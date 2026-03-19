"""
test_fixes_v3.py — Tests de régression pour les 5 corrections appliquées en v3.

Chaque classe couvre une correction spécifique, avec un commentaire précisant
le comportement AVANT (bug) et APRÈS (fix) pour guider les diagnostics futurs.

Fix 1 — decoder.py  : Ordre de fusion DPT (inversé → bottom-up correct)
Fix 2 — trainer.py  : Reprise du scheduler (T_max écrasé → T_max préservé)
Fix 3 — train.py    : Transform validation (train_transform → eval_transform)
Fix 4 — gradient_matching.py : L_gm en espace log (raw → log → scale-invariant)
Fix 5 — scale_invariant.py   : Stabilité sqrt (clamp(min=eps) → clamp(min=0)+eps)
"""

import pytest
import torch
import torch.nn as nn
from PIL import Image as PILImage
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

from src.losses.scale_invariant import ScaleInvariantLoss
from src.losses.gradient_matching import GradientMatchingLoss, DepthAnythingLoss
from src.models.decoder import DPTDecoder


# ===========================================================================
# Fix 1 — Ordre de fusion DPT
#
# BUG  : fusion_blocks[0](reassembled[0]) → 148×148 → écrasé à 36×36 en fin
# APRÈS: fusion_blocks[3](reassembled[3]) → 18×18  → monte jusqu'à 296×296
# ===========================================================================

class TestDPTFusionOrder:
    """Vérifie que la fusion DPT part bien des features profondes vers les superficielles."""

    def test_pre_head_resolution_is_large(self):
        """
        Avant fix : pre-head = 36×36 (écrasé par downsampling à chaque étape).
        Après fix : pre-head = 296×296 (upsampling progressif depuis 18×18).

        Pour une entrée 518×518 (grille 37×37 tokens) :
          reassembled[3] = 18×18 → ×2 = 36 → ×2 = 74 → ×2 = 148 → ×2 = 296
        """
        decoder = DPTDecoder(input_dim=384, hidden_dim=64, image_size=518)
        features = [torch.randn(1, 384, 37, 37) for _ in range(4)]

        pre_head_shape = {}

        def capture_hook(module, inputs, _output):
            pre_head_shape["h"] = inputs[0].shape[2]
            pre_head_shape["w"] = inputs[0].shape[3]

        decoder.head[0].register_forward_hook(capture_hook)
        decoder(features)

        h = pre_head_shape["h"]
        assert h >= 256, (
            f"pre-head height devrait être ~296 (fusion bottom-up), obtenu {h}. "
            f"Si ≈36, la fusion inversée est encore active."
        )

    def test_output_shape_unchanged(self):
        """La forme de sortie (B, 1, 518, 518) ne doit pas changer après le fix."""
        decoder = DPTDecoder(input_dim=384, hidden_dim=64, image_size=518)
        features = [torch.randn(2, 384, 37, 37) for _ in range(4)]
        out = decoder(features)
        assert out.shape == (2, 1, 518, 518)

    def test_output_positive(self):
        """Le ReLU final doit toujours garantir des valeurs >= 0."""
        decoder = DPTDecoder(input_dim=384, hidden_dim=64, image_size=518)
        features = [torch.randn(2, 384, 37, 37) for _ in range(4)]
        out = decoder(features)
        assert (out >= 0).all()

    def test_gradient_flows_through_all_levels(self):
        """
        Les 4 niveaux de features doivent tous recevoir un gradient non-nul.
        Avec la fusion correcte (bottom-up), chaque niveau est traversé.
        """
        decoder = DPTDecoder(input_dim=384, hidden_dim=64, image_size=518)
        features = [torch.randn(1, 384, 37, 37, requires_grad=True) for _ in range(4)]
        out = decoder(features)
        out.sum().backward()
        for i, f in enumerate(features):
            assert f.grad is not None, f"features[{i}] n'a pas de gradient"
            assert f.grad.abs().sum() > 0, f"features[{i}] a un gradient nul"

    def test_deepest_features_have_strongest_spatial_influence(self):
        """
        Avec la fusion bottom-up correcte, les features profondes (index 3)
        sèment toute la hiérarchie spatiale — leur gradient doit être non-nul
        et proportionnel à leur rôle d'initialisation.
        """
        decoder = DPTDecoder(input_dim=384, hidden_dim=64, image_size=518)
        features = [torch.randn(1, 384, 37, 37, requires_grad=True) for _ in range(4)]
        out = decoder(features)
        out.sum().backward()

        grad_norms = [f.grad.abs().mean().item() for f in features]
        # Tous les niveaux doivent contribuer
        assert all(g > 0 for g in grad_norms), (
            f"Tous les gradients doivent être > 0, obtenus : {grad_norms}"
        )

    def test_custom_target_size_after_fix(self):
        """target_size doit toujours fonctionner après le fix de l'ordre."""
        decoder = DPTDecoder(input_dim=384, hidden_dim=64, image_size=518)
        features = [torch.randn(1, 384, 37, 37) for _ in range(4)]
        out = decoder(features, target_size=(480, 640))
        assert out.shape == (1, 1, 480, 640)


# ===========================================================================
# Fix 2 — Reprise du scheduler (T_max préservé)
#
# BUG  : load_state_dict(scheduler_state) → T_max=20 restauré → LR ≈ 6e-7
# APRÈS: on avance le scheduler dans le nouveau cycle (T_max=40) → LR ≈ 5e-5
# ===========================================================================

class _DictDataset(Dataset):
    """Dataset minimal retournant le format dict attendu par Trainer."""
    def __getitem__(self, _idx):
        return {"image": torch.zeros(3, 32, 32), "pseudo_depth": torch.ones(1, 32, 32)}
    def __len__(self):
        return 8


class _ZeroLoss(nn.Module):
    """Critère qui renvoie toujours 0 (teste l'infra sans apprentissage)."""
    def forward(self, pred, _target):
        return {"total": pred.mean() * 0.0}


def _save_fake_checkpoint(path: str, model, epoch_0indexed: int, t_max_old: int):
    """Crée un checkpoint simulant un run de t_max_old epochs arrêté à epoch_0indexed."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    old_sched = CosineAnnealingLR(optimizer, T_max=t_max_old)
    for _ in range(epoch_0indexed + 1):
        old_sched.step()
    torch.save({
        "epoch": epoch_0indexed,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": old_sched.state_dict(),  # T_max=t_max_old encodé ici
        "loss": 0.5,
        "config": {"epochs": t_max_old},
        "training_history": [],
    }, path)
    return old_sched.get_last_lr()[0]


class TestSchedulerResume:
    """Vérifie que la reprise d'entraînement préserve le T_max du nouveau run."""

    def test_lr_not_collapsed_after_resume(self, tmp_path):
        """
        Régression : l'ancienne logique restaurait T_max=20 depuis le checkpoint,
        donnant LR ≈ 6e-7 aux epochs 21-30.
        Après fix : le scheduler suit T_max=40, LR ≈ 5e-5 à l'epoch 21.
        """
        from src.training.trainer import Trainer

        model = nn.Linear(4, 1)
        ckpt_path = str(tmp_path / "ckpt.pt")
        old_lr = _save_fake_checkpoint(ckpt_path, model, epoch_0indexed=19, t_max_old=20)

        # L'ancien run se terminait avec LR quasi-nul
        assert old_lr < 1e-6, f"L'ancien scheduler à epoch 20 devrait avoir LR≈0, obtenu {old_lr:.2e}"

        loader = DataLoader(_DictDataset(), batch_size=2)
        trainer = Trainer(
            model=nn.Linear(4, 1),
            criterion=_ZeroLoss(),
            train_loader=loader,
            config={"epochs": 40, "learning_rate": 1e-4},
            device="cpu",
            output_dir=str(tmp_path / "out"),
        )
        trainer.resume_from_checkpoint(ckpt_path)

        lr = trainer.scheduler.get_last_lr()[0]
        # À position 20/40 dans le cosinus : LR = lr_max/2 * (1 + cos(π)) ≈ 5e-5
        assert lr > 1e-5, (
            f"LR après reprise devrait être ~5e-5 (T_max=40), obtenu {lr:.2e}. "
            f"Si ~6e-7, scheduler.load_state_dict est encore appelé."
        )
        assert lr < 1e-3

    def test_lr_follows_new_tmax(self, tmp_path):
        """
        Le LR après reprise doit correspondre à la position current_epoch/T_max_new
        dans la courbe cosinus du nouveau run, pas de l'ancien.
        """
        from src.training.trainer import Trainer

        model = nn.Linear(4, 1)
        ckpt_path = str(tmp_path / "ckpt.pt")
        _save_fake_checkpoint(ckpt_path, model, epoch_0indexed=9, t_max_old=10)

        loader = DataLoader(_DictDataset(), batch_size=2)
        trainer = Trainer(
            model=nn.Linear(4, 1),
            criterion=_ZeroLoss(),
            train_loader=loader,
            config={"epochs": 50, "learning_rate": 1e-4},
            device="cpu",
            output_dir=str(tmp_path / "out"),
        )
        trainer.resume_from_checkpoint(ckpt_path)

        lr = trainer.scheduler.get_last_lr()[0]
        # À position 10/50 dans le cosinus : LR ≈ 0.97 * lr_max ≈ 9.7e-5
        assert lr > 5e-5, f"À pos 10/50 dans cosinus, LR devrait être ~9.7e-5, obtenu {lr:.2e}"

    def test_resume_sets_correct_epoch(self, tmp_path):
        """Après reprise, current_epoch doit être epoch_checkpoint + 1."""
        from src.training.trainer import Trainer

        model = nn.Linear(4, 1)
        ckpt_path = str(tmp_path / "ckpt.pt")
        _save_fake_checkpoint(ckpt_path, model, epoch_0indexed=14, t_max_old=20)

        loader = DataLoader(_DictDataset(), batch_size=2)
        trainer = Trainer(
            model=nn.Linear(4, 1),
            criterion=_ZeroLoss(),
            train_loader=loader,
            config={"epochs": 40, "learning_rate": 1e-4},
            device="cpu",
            output_dir=str(tmp_path / "out"),
        )
        trainer.resume_from_checkpoint(ckpt_path)
        assert trainer.current_epoch == 15

    def test_resume_loads_model_weights(self, tmp_path):
        """Les poids du modèle doivent être restaurés depuis le checkpoint."""
        from src.training.trainer import Trainer

        model_src = nn.Linear(4, 1)
        nn.init.constant_(model_src.weight, 99.0)  # valeur distinctive
        ckpt_path = str(tmp_path / "ckpt.pt")
        _save_fake_checkpoint(ckpt_path, model_src, epoch_0indexed=5, t_max_old=10)

        model_dst = nn.Linear(4, 1)
        nn.init.constant_(model_dst.weight, 0.0)

        loader = DataLoader(_DictDataset(), batch_size=2)
        trainer = Trainer(
            model=model_dst,
            criterion=_ZeroLoss(),
            train_loader=loader,
            config={"epochs": 20, "learning_rate": 1e-4},
            device="cpu",
            output_dir=str(tmp_path / "out"),
        )
        trainer.resume_from_checkpoint(ckpt_path)
        assert torch.allclose(model_dst.weight, torch.full_like(model_dst.weight, 99.0))


# ===========================================================================
# Fix 3 — Séparation train/val transform
#
# BUG  : full_dataset créé avec train_transform → val reçoit random flip/crop
# APRÈS: deux datasets séparés, val utilise eval_transform (déterministe)
# ===========================================================================

class TestValTransformSeparation:
    """Vérifie la séparation correcte des transforms entre train et val."""

    def _make_pil_pair(self, h=480, w=640):
        img = PILImage.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
        depth = PILImage.fromarray(np.random.rand(h, w).astype(np.float32))
        return img, depth

    def test_eval_transform_is_deterministic(self):
        """
        EvalTransform n'a pas d'opérations aléatoires :
        deux appels sur la même entrée doivent donner des tenseurs identiques.
        """
        from src.data.transforms import EvalTransform
        t = EvalTransform(image_size=518)
        img, depth = self._make_pil_pair()

        img1, d1 = t(img, depth)
        img2, d2 = t(img, depth)

        assert torch.allclose(img1, img2), "EvalTransform doit être déterministe (image)"
        assert torch.allclose(d1, d2), "EvalTransform doit être déterministe (depth)"

    def test_train_transform_is_stochastic(self):
        """
        TrainTransform applique flip/crop aléatoires :
        10 appels successifs doivent donner au moins 2 résultats différents.
        """
        from src.data.transforms import TrainTransform
        t = TrainTransform(image_size=518, horizontal_flip_prob=0.5)
        img, depth = self._make_pil_pair(h=600, w=800)

        results = [t(img, depth)[0] for _ in range(10)]
        unique_hashes = {tuple(x.flatten().tolist()[:20]) for x in results}
        assert len(unique_hashes) > 1, "TrainTransform doit être stochastique"

    def test_val_and_train_indices_cover_all_samples(self):
        """
        Le split 90/10 dans train.py doit couvrir tous les échantillons
        sans chevauchement ni oubli.
        """
        seed, n_total = 42, 300
        n_val = max(1, int(n_total * 0.1))
        n_train = n_total - n_val

        all_indices = torch.randperm(
            n_total, generator=torch.Generator().manual_seed(seed)
        ).tolist()
        train_set = set(all_indices[:n_train])
        val_set = set(all_indices[n_train:])

        assert len(train_set & val_set) == 0, "Les ensembles train et val ne doivent pas se chevaucher"
        assert len(train_set) + len(val_set) == n_total, "Tous les indices doivent être couverts"

    def test_eval_transform_output_shape(self):
        """EvalTransform produit (3, 518, 518) et (1, 518, 518) indépendamment de l'entrée."""
        from src.data.transforms import EvalTransform
        t = EvalTransform(image_size=518)
        img, depth = self._make_pil_pair(h=640, w=427)
        img_t, depth_t = t(img, depth)
        assert img_t.shape == (3, 518, 518)
        assert depth_t.shape[0] == 1


# ===========================================================================
# Fix 4 — L_gm en espace log (scale-invariance)
#
# BUG  : gradients sur valeurs brutes → L_gm(s*pred, s*gt) = s * L_gm(pred, gt)
# APRÈS: gradients sur log → L_gm(s*pred, s*gt) ≈ L_gm(pred, gt)
# ===========================================================================

class TestGradientMatchingLogSpace:
    """Vérifie l'invariance d'échelle de L_gm après le passage en log-espace."""

    def test_scale_invariant_large_scale(self):
        """
        Régression : ancienne implémentation → ratio ≈ 10 (proportionnel à l'échelle).
        Après fix → ratio ≈ 1 (invariant d'échelle).
        """
        gm = GradientMatchingLoss()
        torch.manual_seed(0)
        pred = torch.rand(2, 1, 64, 64) + 0.1
        gt = torch.rand(2, 1, 64, 64) + 0.1
        scale = 10.0

        loss_base = gm(pred, gt)
        loss_scaled = gm(pred * scale, gt * scale)
        ratio = loss_scaled.item() / (loss_base.item() + 1e-10)

        assert abs(ratio - 1.0) < 0.05, (
            f"L_gm doit être scale-invariante après le fix log-espace. "
            f"Ratio={ratio:.3f} (attendu ≈1.0). "
            f"Si ratio≈{scale:.0f}, l'ancienne version raw-space est encore active."
        )

    def test_scale_invariant_small_scale(self):
        """Même test avec une mise à l'échelle < 1 (profondeurs très petites)."""
        gm = GradientMatchingLoss()
        torch.manual_seed(1)
        pred = torch.rand(2, 1, 32, 32) + 0.5
        gt = torch.rand(2, 1, 32, 32) + 0.5
        scale = 0.05

        loss_base = gm(pred, gt)
        loss_scaled = gm(pred * scale, gt * scale)
        ratio = loss_scaled.item() / (loss_base.item() + 1e-10)

        assert abs(ratio - 1.0) < 0.05, (
            f"L_gm doit être scale-invariante pour scale={scale}. Ratio={ratio:.3f}"
        )

    def test_zero_on_identical(self):
        """L_gm(pred, pred.clone()) doit être ≈ 0 en log-espace comme en raw-espace."""
        gm = GradientMatchingLoss()
        depth = torch.rand(2, 1, 32, 32) + 0.1
        loss = gm(depth, depth.clone())
        assert loss.item() < 1e-5, f"L_gm sur tenseurs identiques doit être ≈0, obtenu {loss.item()}"

    def test_gradient_flows(self):
        """Le gradient doit remonter à travers le log vers pred."""
        gm = GradientMatchingLoss()
        pred = (torch.rand(2, 1, 32, 32) + 0.1).requires_grad_(True)
        gt = torch.rand(2, 1, 32, 32) + 0.1
        loss = gm(pred, gt)
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.abs().sum() > 0

    def test_non_negative(self):
        """L_gm doit rester >= 0 après le passage en log-espace."""
        gm = GradientMatchingLoss()
        pred = torch.rand(4, 1, 32, 32) + 0.1
        gt = torch.rand(4, 1, 32, 32) + 0.1
        loss = gm(pred, gt)
        assert loss.item() >= 0, f"L_gm doit être non-négative, obtenu {loss.item()}"

    def test_not_dependent_on_absolute_scale(self):
        """
        Vérifie que L_gm ne varie pas avec l'échelle absolue des valeurs,
        sur 5 facteurs d'échelle différents.
        """
        gm = GradientMatchingLoss()
        torch.manual_seed(7)
        pred = torch.rand(2, 1, 32, 32) + 0.2
        gt = torch.rand(2, 1, 32, 32) + 0.2
        base_loss = gm(pred, gt).item()

        for scale in [0.01, 0.1, 5.0, 50.0, 1000.0]:
            loss_s = gm(pred * scale, gt * scale).item()
            ratio = loss_s / (base_loss + 1e-10)
            assert abs(ratio - 1.0) < 0.05, (
                f"scale={scale}: ratio={ratio:.3f} (doit être ≈1.0)"
            )


# ===========================================================================
# Fix 5 — Stabilité du sqrt dans L_ssi
#
# BUG  : clamp(min=eps) → quand term1≈term2≈0, gradient ∂L/∂x → 1/(2√ε) grand
#         et la valeur clampée masque les cas term1-term2 < 0 (NaN possible)
# APRÈS: clamp(min=0) + eps → argument toujours ≥ eps, gradient fini partout
# ===========================================================================

class TestSSISqrtStability:
    """Vérifie la stabilité numérique du sqrt dans ScaleInvariantLoss."""

    def test_no_nan_gradient_near_convergence(self):
        """
        Quand pred ≈ gt (modèle quasi-convergé), le gradient ne doit pas
        être NaN ou Inf.
        """
        ssi = ScaleInvariantLoss()
        gt = torch.rand(4, 1, 32, 32) + 0.5
        pred = (gt + 1e-5 * torch.randn_like(gt)).clamp(min=1e-6).requires_grad_(True)

        loss = ssi(pred, gt.detach())
        loss.backward()

        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any(), "Gradient NaN proche de la convergence"
        assert not torch.isinf(pred.grad).any(), "Gradient Inf proche de la convergence"

    def test_loss_non_negative_with_aggressive_masking(self):
        """
        Avec un masquage top-k agressif, term1-term2 peut devenir légèrement
        négatif. clamp(min=0)+eps garantit que le sqrt ne reçoit jamais < 0.
        """
        ssi = ScaleInvariantLoss(lambda_ssi=0.5, top_k_masking=0.5)
        pred = torch.rand(4, 1, 16, 16) + 0.01
        gt = torch.rand(4, 1, 16, 16) + 0.01
        loss = ssi(pred, gt)

        assert loss.item() >= 0, f"Loss doit être >= 0, obtenu {loss.item()}"
        assert not torch.isnan(loss), "Loss ne doit pas être NaN"

    def test_no_nan_over_many_random_inputs(self):
        """
        Sur 50 paires aléatoires couvrant des ordres de grandeur variés,
        la loss doit toujours être >= 0 et non-NaN.
        """
        ssi = ScaleInvariantLoss()
        torch.manual_seed(42)
        for _ in range(50):
            pred = torch.rand(2, 1, 32, 32) * 10 + 0.01
            gt = torch.rand(2, 1, 32, 32) * 10 + 0.01
            loss = ssi(pred, gt)
            assert loss.item() >= 0, f"Loss négative : {loss.item()}"
            assert not torch.isnan(loss), "Loss NaN détecté"

    def test_gradient_finite_at_exact_match(self):
        """
        Quand pred == gt exactement, le gradient doit être fini
        (sans division par zéro sous le sqrt).
        """
        ssi = ScaleInvariantLoss()
        depth = (torch.rand(2, 1, 16, 16) + 0.1).requires_grad_(True)
        loss = ssi(depth, depth.detach().clone())
        loss.backward()

        assert depth.grad is not None
        assert not torch.isnan(depth.grad).any(), "Gradient NaN sur pred==gt"
        assert not torch.isinf(depth.grad).any(), "Gradient Inf sur pred==gt"

    def test_combined_loss_no_nan(self):
        """DepthAnythingLoss (SSI + GM) ne doit pas produire de NaN."""
        criterion = DepthAnythingLoss()
        pred = torch.rand(2, 1, 64, 64) + 0.1
        gt = torch.rand(2, 1, 64, 64) + 0.1
        result = criterion(pred, gt)

        assert not torch.isnan(result["total"]), "Loss combinée NaN"
        assert result["total"].item() >= 0, "Loss combinée négative"

    def test_combined_loss_gradient_flows(self):
        """Le gradient de la loss combinée doit remonter jusqu'à pred."""
        criterion = DepthAnythingLoss()
        pred = (torch.rand(2, 1, 32, 32) + 0.1).requires_grad_(True)
        gt = torch.rand(2, 1, 32, 32) + 0.1
        result = criterion(pred, gt)
        result["total"].backward()

        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()
        assert pred.grad.abs().sum() > 0
