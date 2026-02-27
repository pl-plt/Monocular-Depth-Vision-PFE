"""
train.py — Phase 4.2-4.3 : Entraînement du Student.

Deux modes d'entraînement :
1. Initial (Phase 4.2) : Entraîner sur 50k images, 20 epochs
2. Scale-up (Phase 4.3) : Fine-tuner sur 200k images, 10-15 epochs

Hyperparamètres recommandés :
    batch_size = 16
    learning_rate = 1e-4
    optimizer = AdamW (weight_decay=0.01)
    scheduler = CosineAnnealingLR
    gradient_clip = 1.0

Usage :
    # Entraînement initial (50k)
    python scripts/train.py \
        --images_dir datasets/real_unlabeled/sa1b/images \
        --pseudo_labels_dir outputs/pseudo_labels \
        --output_dir outputs \
        --epochs 20 --lr 1e-4 --batch_size 16

    # Scale-up (200k), reprendre depuis checkpoint
    python scripts/train.py \
        --images_dir datasets/real_unlabeled/sa1b_200k/images \
        --pseudo_labels_dir outputs/pseudo_labels_200k \
        --resume outputs/checkpoints/best_model.pt \
        --epochs 15 --lr 5e-5

    # Overfitting test (10 images, sanity check)
    python scripts/train.py \
        --images_dir datasets/synthetic/hypersim/images \
        --pseudo_labels_dir datasets/synthetic/hypersim/depth \
        --max_samples 10 --epochs 100 --lr 1e-3
"""

import sys
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.student import StudentModel
from src.losses import DepthAnythingLoss
from src.data.datasets import PseudoLabeledDataset
from src.data.transforms import get_train_transforms, get_eval_transforms
from src.training.trainer import Trainer
from src.utils.helpers import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 4.2-4.3 — Entraînement Student")

    # Données
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Répertoire d'images réelles")
    parser.add_argument("--pseudo_labels_dir", type=str, required=True,
                        help="Répertoire de pseudo-labels")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Nombre max d'échantillons (pour debug/overfitting test)")

    # Modèle
    parser.add_argument("--backbone", type=str, default="dinov2_vits14",
                        help="Backbone DINOv2 du Student")
    parser.add_argument("--image_size", type=int, default=518,
                        help="Taille des images")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Geler le backbone (fine-tune decoder seul)")

    # Entraînement
    parser.add_argument("--epochs", type=int, default=20, help="Nombre d'epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--gradient_clip", type=float, default=1.0, help="Max norm gradient clipping")

    # Loss
    parser.add_argument("--lambda_ssi", type=float, default=0.5, help="Lambda pour L_ssi")
    parser.add_argument("--alpha_gm", type=float, default=0.5, help="Poids de L_gm")
    parser.add_argument("--top_k_masking", type=float, default=0.1, help="Top-K% masking")

    # Reprise
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint pour reprise")

    # Sortie
    parser.add_argument("--output_dir", type=str, default="outputs", help="Répertoire de sortie")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire")
    parser.add_argument("--num_workers", type=int, default=8, help="Workers DataLoader")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    print("=" * 60)
    print("Phase 4 : Entraînement du Student")
    print("=" * 60)

    # 1. Modèle Student
    print("\n--- Initialisation du Student ---")
    student = StudentModel(
        backbone_name=args.backbone,
        pretrained_backbone=True,
        freeze_backbone=args.freeze_backbone,
        image_size=args.image_size,
    )
    params = student.count_parameters()
    print(f"  Total      : {params['total_M']:.1f}M")
    print(f"  Trainable  : {params['trainable_M']:.1f}M")

    # 2. Loss
    criterion = DepthAnythingLoss(
        lambda_ssi=args.lambda_ssi,
        alpha_gm=args.alpha_gm,
        top_k_masking=args.top_k_masking,
    )

    # 3. Dataset
    print("\n--- Chargement des données ---")
    train_transform = get_train_transforms(image_size=args.image_size)
    eval_transform = get_eval_transforms(image_size=args.image_size)

    full_dataset = PseudoLabeledDataset(
        image_dir=args.images_dir,
        pseudo_label_dir=args.pseudo_labels_dir,
        transform=train_transform,
        max_samples=args.max_samples,
    )

    # Split train/val (90/10)
    n_val = max(1, int(len(full_dataset) * 0.1))
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    print(f"  Train : {n_train} images")
    print(f"  Val   : {n_val} images")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, args.num_workers),
        pin_memory=True,
    )

    # 4. Configuration
    train_config = {
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "gradient_clip_max_norm": args.gradient_clip,
    }

    # 5. Trainer
    trainer = Trainer(
        model=student,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=str(device),
        output_dir=args.output_dir,
    )

    # Reprise depuis checkpoint
    if args.resume:
        print(f"\n--- Reprise depuis checkpoint : {args.resume} ---")
        trainer.resume_from_checkpoint(args.resume)

    # 6. Entraînement
    print("\n--- Début de l'entraînement ---")
    results = trainer.train()

    print("\n✅ Entraînement terminé.")
    print(f"   Meilleur modèle : {args.output_dir}/checkpoints/best_model.pt")


if __name__ == "__main__":
    main()
