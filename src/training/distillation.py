"""
distillation.py — Pipeline de distillation Teacher → Student.

Orchestre le processus complet :
1. Charger le Teacher (DINOv2-Giant, figé)
2. Générer les pseudo-labels sur les images réelles
3. Entraîner le Student sur les pseudo-labels

La distillation est au niveau de la prédiction (pseudo-labels),
pas au niveau des features. C'est plus sûr quand il y a un grand
écart de taille entre Teacher et Student.

Ref: Section 4 et 8 du papier
"""

import torch
from pathlib import Path
from typing import Optional

from ..models.teacher import TeacherModel
from ..models.student import StudentModel
from ..losses import DepthAnythingLoss
from ..data.datasets import PseudoLabeledDataset
from ..data.transforms import get_train_transforms
from .trainer import Trainer
from .pseudo_labels import PseudoLabelGenerator


class DistillationPipeline:
    """
    Pipeline complet de distillation Teacher → Student.

    Étapes :
    1. Charger le Teacher pré-entraîné (sur synthétiques)
    2. Générer pseudo-labels sur images réelles (Phase 4.1)
    3. Entraîner le Student sur pseudo-labels (Phase 4.2-4.3)

    Args:
        teacher_weights: Chemin vers les poids du Teacher.
        student_backbone: Variante DINOv2 pour le Student.
        image_size: Taille des images.
        device: Device de calcul.
        output_dir: Répertoire de sortie.
    """

    def __init__(
        self,
        teacher_weights: str,
        student_backbone: str = "dinov2_vits14",
        image_size: int = 518,
        device: str = "cuda",
        output_dir: str = "outputs",
    ):
        self.device = device
        self.image_size = image_size
        self.output_dir = Path(output_dir)

        # Charger le Teacher (figé)
        print("Chargement du Teacher...")
        self.teacher = TeacherModel(
            pretrained_weights=teacher_weights,
            image_size=image_size,
        ).to(device)

        # Créer le Student
        print("Initialisation du Student...")
        self.student = StudentModel(
            backbone_name=student_backbone,
            pretrained_backbone=True,
            image_size=image_size,
        ).to(device)

        # Afficher les paramètres
        student_params = self.student.count_parameters()
        print(f"Student : {student_params['total_M']:.1f}M params "
              f"({student_params['trainable_M']:.1f}M entraînables)")

    def step1_generate_pseudo_labels(
        self,
        unlabeled_image_dir: str,
        pseudo_label_dir: Optional[str] = None,
        batch_size: int = 16,
    ):
        """
        Étape 1 : Générer les pseudo-labels avec le Teacher.

        Calcul de temps estimé :
        - 50k images × 0.2 sec/image ≈ 2.8 heures
        - 200k images × 0.2 sec/image ≈ 11 heures

        Args:
            unlabeled_image_dir: Répertoire d'images réelles.
            pseudo_label_dir: Répertoire de sortie.
            batch_size: Taille des batches.
        """
        if pseudo_label_dir is None:
            pseudo_label_dir = str(self.output_dir / "pseudo_labels")

        generator = PseudoLabelGenerator(
            teacher=self.teacher,
            device=self.device,
            output_dir=pseudo_label_dir,
        )

        generator.generate(
            image_dir=unlabeled_image_dir,
            batch_size=batch_size,
            image_size=self.image_size,
        )

    def step2_train_student(
        self,
        image_dir: str,
        pseudo_label_dir: str,
        train_config: Optional[dict] = None,
    ):
        """
        Étape 2 : Entraîner le Student sur les pseudo-labels.

        Args:
            image_dir: Répertoire d'images réelles.
            pseudo_label_dir: Répertoire de pseudo-labels.
            train_config: Configuration d'entraînement.
        """
        # Dataset pseudo-labélisé
        transform = get_train_transforms(image_size=self.image_size)
        dataset = PseudoLabeledDataset(
            image_dir=image_dir,
            pseudo_label_dir=pseudo_label_dir,
            transform=transform,
        )

        # Split train/val
        n_val = int(len(dataset) * 0.1)
        n_train = len(dataset) - n_val
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_val]
        )

        # DataLoaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_config.get("batch_size", 16) if train_config else 16,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=2,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=train_config.get("batch_size", 16) if train_config else 16,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # Loss
        criterion = DepthAnythingLoss()

        # Trainer
        trainer = Trainer(
            model=self.student,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            config=train_config,
            device=self.device,
            output_dir=str(self.output_dir),
        )

        # Lancer l'entraînement
        results = trainer.train()
        return results
