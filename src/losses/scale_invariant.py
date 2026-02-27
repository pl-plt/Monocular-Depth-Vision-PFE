"""
scale_invariant.py — Scale-and-Shift Invariant Loss (L_ssi).

La depth prédite peut être à une échelle et un décalage différents
de la ground truth (surtout en profondeur relative). Cette loss
normalise les prédictions avant de calculer l'erreur.

Formule :
    L_ssi = sqrt( (1/n) * Σ(d_i - d_i*)² - (λ/n²) * (Σ(d_i - d_i*))² )

    où d_i = log(pred_i), d_i* = log(gt_i)
    λ ∈ [0, 1] contrôle l'invariance à l'échelle (λ=0.5 recommandé)

Inclut aussi le masquage top-10% : on ignore les 10% de pixels
avec les erreurs les plus élevées pour robustifier l'entraînement.

Ref: Section 5 du papier + Eigen et al. (2014)
"""

import torch
import torch.nn as nn


class ScaleInvariantLoss(nn.Module):
    """
    Scale-and-Shift Invariant Loss pour l'estimation de profondeur.

    Args:
        lambda_ssi: Coefficient λ d'invariance à l'échelle (défaut 0.5).
        use_log: Si True, travaille en espace log-depth.
        top_k_masking: Fraction de pixels à masquer (ex: 0.1 = top 10%).
        eps: Epsilon pour stabilité numérique.
    """

    def __init__(
        self,
        lambda_ssi: float = 0.5,
        use_log: bool = True,
        top_k_masking: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.lambda_ssi = lambda_ssi
        self.use_log = use_log
        self.top_k_masking = top_k_masking
        self.eps = eps

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Calcul de la loss L_ssi.

        Args:
            prediction: Depth maps prédites [B, 1, H, W].
            target: Depth maps ground truth / pseudo-labels [B, 1, H, W].
            valid_mask: Masque de pixels valides [B, 1, H, W] (optionnel).

        Returns:
            Loss scalaire.
        """
        # Aplatir les dimensions spatiales
        pred = prediction.flatten(1)  # [B, N]
        tgt = target.flatten(1)       # [B, N]

        if valid_mask is not None:
            mask = valid_mask.flatten(1).bool()  # [B, N]
        else:
            mask = (tgt > self.eps)  # Ignorer les pixels sans profondeur

        # Appliquer en espace log si demandé
        if self.use_log:
            pred = torch.log(pred.clamp(min=self.eps))
            tgt = torch.log(tgt.clamp(min=self.eps))

        # Calculer les différences
        diff = pred - tgt  # [B, N]

        # Appliquer le masque de validité
        diff = diff * mask.float()

        # Masquage top-K% : ignorer les pixels avec les plus grandes erreurs
        if self.top_k_masking > 0:
            diff, mask = self._apply_top_k_masking(diff, mask)

        # Nombre de pixels valides par image
        n = mask.sum(dim=1).clamp(min=1).float()  # [B]

        # Loss scale-invariant
        # Terme 1 : (1/n) * Σ(d_i - d_i*)²
        term1 = (diff ** 2).sum(dim=1) / n

        # Terme 2 : (λ/n²) * (Σ(d_i - d_i*))²
        term2 = self.lambda_ssi * (diff.sum(dim=1) ** 2) / (n ** 2)

        # Loss finale : sqrt(term1 - term2)
        loss = torch.sqrt((term1 - term2).clamp(min=self.eps))

        return loss.mean()

    def _apply_top_k_masking(
        self,
        diff: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple:
        """
        Masque les top-K% pixels avec les erreurs les plus élevées.

        Stratégie du papier : ignorer les 10% de pixels avec les
        erreurs les plus grandes pour éviter que des outliers
        ne dominent le gradient.

        Args:
            diff: Différences pred-target [B, N].
            mask: Masque courant [B, N].

        Returns:
            (diff_masked, new_mask) avec les top-K% erreurs mises à zéro.
        """
        B, N = diff.shape
        abs_diff = diff.abs()

        # Nombre de pixels valides
        n_valid = mask.sum(dim=1, keepdim=True)  # [B, 1]
        n_to_mask = (n_valid * self.top_k_masking).long()  # [B, 1]

        # Trouver le seuil pour chaque image
        # On trie les erreurs et on garde le seuil au percentile (1 - top_k)
        sorted_diff, _ = abs_diff.sort(dim=1, descending=True)

        # Pour chaque image, le seuil est la (n_to_mask)-ème plus grande erreur
        threshold = sorted_diff.gather(1, n_to_mask.clamp(max=N - 1))  # [B, 1]

        # Masquer les pixels au-dessus du seuil
        top_k_mask = abs_diff < threshold
        new_mask = mask & top_k_mask

        # Remettre à zéro les pixels masqués
        diff = diff * new_mask.float()

        return diff, new_mask
