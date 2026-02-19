"""
visualization.py — Visualisation des depth maps et comparaisons.

Outils de visualisation :
- Depth maps individuelles (colormap viridis)
- Grilles comparatives : Image | GT | Prédiction | DAv2 officiel
- Best/worst cases analysis
- Histogrammes de distribution de profondeur

Ref: Phase 0 (visualisation baseline) + Phase 5 (analyse qualitative)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import List, Optional, Tuple


class DepthVisualizer:
    """
    Visualiseur de depth maps pour analyse qualitative.

    Args:
        colormap: Colormap matplotlib pour les depth maps.
        output_dir: Répertoire de sauvegarde des images.
        figsize: Taille des figures par défaut.
    """

    def __init__(
        self,
        colormap: str = "viridis",
        output_dir: str = "outputs/visualizations",
        figsize: Tuple[int, int] = (16, 8),
    ):
        self.colormap = colormap
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize

    def visualize_single(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        title: str = "",
        save_path: Optional[str] = None,
    ):
        """
        Affiche une image et sa depth map côte à côte.

        Args:
            image: Image RGB [H, W, 3].
            depth: Depth map [H, W].
            title: Titre de la figure.
            save_path: Chemin de sauvegarde (optionnel).
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)

        axes[0].imshow(image)
        axes[0].set_title("Image RGB")
        axes[0].axis("off")

        im = axes[1].imshow(depth, cmap=self.colormap)
        axes[1].set_title("Depth Map")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        if title:
            fig.suptitle(title, fontsize=14)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def visualize_comparison(
        self,
        image: np.ndarray,
        ground_truth: np.ndarray,
        prediction: np.ndarray,
        official: Optional[np.ndarray] = None,
        title: str = "",
        save_path: Optional[str] = None,
    ):
        """
        Grille comparative : Image | GT | Prédiction | (Officiel).

        Args:
            image: Image RGB [H, W, 3].
            ground_truth: Depth GT [H, W].
            prediction: Notre prédiction [H, W].
            official: Prédiction du modèle officiel (optionnel).
            title: Titre.
            save_path: Chemin de sauvegarde.
        """
        n_cols = 4 if official is not None else 3
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

        axes[0].imshow(image)
        axes[0].set_title("Image")
        axes[0].axis("off")

        axes[1].imshow(ground_truth, cmap=self.colormap)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(prediction, cmap=self.colormap)
        axes[2].set_title("Notre modèle")
        axes[2].axis("off")

        if official is not None:
            axes[3].imshow(official, cmap=self.colormap)
            axes[3].set_title("DAv2 officiel")
            axes[3].axis("off")

        if title:
            fig.suptitle(title, fontsize=14)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def visualize_error_map(
        self,
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        title: str = "",
        save_path: Optional[str] = None,
    ):
        """
        Affiche la carte d'erreur (différence absolue).

        Args:
            prediction: Depth prédite [H, W].
            ground_truth: Depth GT [H, W].
        """
        error = np.abs(prediction - ground_truth)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].imshow(ground_truth, cmap=self.colormap)
        axes[0].set_title("Ground Truth")
        axes[0].axis("off")

        axes[1].imshow(prediction, cmap=self.colormap)
        axes[1].set_title("Prédiction")
        axes[1].axis("off")

        im = axes[2].imshow(error, cmap="hot")
        axes[2].set_title("Erreur absolue")
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], fraction=0.046)

        if title:
            fig.suptitle(title, fontsize=14)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_training_history(
        self,
        history: List[dict],
        save_path: Optional[str] = None,
    ):
        """
        Plot les courbes d'entraînement (loss, lr).

        Args:
            history: Liste de dicts avec train_loss, val_loss, lr.
            save_path: Chemin de sauvegarde.
        """
        epochs = [h["epoch"] for h in history]
        train_loss = [h.get("train_loss", None) for h in history]
        val_loss = [h.get("val_loss", None) for h in history]
        lr = [h.get("lr", None) for h in history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss
        ax1.plot(epochs, train_loss, "b-", label="Train")
        if any(v is not None for v in val_loss):
            ax1.plot(epochs, val_loss, "r-", label="Validation")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Courbe de Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Learning rate
        if any(v is not None for v in lr):
            ax2.plot(epochs, lr, "g-")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Learning Rate")
            ax2.set_title("Learning Rate Schedule")
            ax2.set_yscale("log")
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def create_best_worst_grid(
        self,
        images: List[np.ndarray],
        predictions: List[np.ndarray],
        ground_truths: List[np.ndarray],
        errors: List[float],
        n_examples: int = 10,
        save_path: Optional[str] = None,
    ):
        """
        Crée une grille des N meilleurs et N pires cas.

        Args:
            images: Liste d'images RGB.
            predictions: Liste de depth prédites.
            ground_truths: Liste de depth GT.
            errors: Liste d'erreurs (AbsRel) par image.
            n_examples: Nombre d'exemples best/worst.
        """
        sorted_indices = np.argsort(errors)
        best_indices = sorted_indices[:n_examples]
        worst_indices = sorted_indices[-n_examples:][::-1]

        fig, axes = plt.subplots(
            n_examples * 2, 3, figsize=(15, 4 * n_examples * 2)
        )

        for row, idx in enumerate(list(best_indices) + list(worst_indices)):
            category = "BEST" if row < n_examples else "WORST"
            rank = row if row < n_examples else row - n_examples

            axes[row, 0].imshow(images[idx])
            axes[row, 0].set_title(f"{category} #{rank+1} — Image")
            axes[row, 0].axis("off")

            axes[row, 1].imshow(predictions[idx], cmap=self.colormap)
            axes[row, 1].set_title(f"Prédiction (err={errors[idx]:.4f})")
            axes[row, 1].axis("off")

            axes[row, 2].imshow(ground_truths[idx], cmap=self.colormap)
            axes[row, 2].set_title("Ground Truth")
            axes[row, 2].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
