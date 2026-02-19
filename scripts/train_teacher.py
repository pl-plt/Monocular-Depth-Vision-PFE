"""
train_teacher.py — Phase 3 : Entraînement du Teacher sur données synthétiques.

Le Teacher utilise DINOv2-Giant (backbone figé, 1.1B params)
avec un décodeur DPT entraîné sur des images synthétiques
qui possèdent des ground truth de profondeur précises.

Architecture :
    DINOv2-Giant (frozen) → DPT Decoder (trainable) → Depth map

Datasets synthétiques (Section A du papier) :
    - Hypersim (indoor)
    - Virtual KITTI 2 (outdoor)

Hyperparamètres recommandés :
    batch_size = 4-8 (Giant model = beaucoup de VRAM)
    learning_rate = 1e-4
    optimizer = AdamW (weight_decay=0.01)
    scheduler = CosineAnnealingLR
    epochs = 20-30

Usage :
    # Entraînement Teacher sur Hypersim
    python scripts/train_teacher.py \
        --dataset_dir datasets/synthetic/hypersim \
        --epochs 25 --lr 1e-4 --batch_size 4

    # Reprendre depuis un checkpoint
    python scripts/train_teacher.py \
        --dataset_dir datasets/synthetic/hypersim \
        --resume outputs/teacher/checkpoints/best_model.pt

    # Overfitting test (10 images, sanity check)
    python scripts/train_teacher.py \
        --dataset_dir datasets/synthetic/hypersim \
        --max_samples 10 --epochs 100 --lr 1e-3
"""

import sys
import os
import argparse
import traceback
from pathlib import Path

import torch

# Forcer le flush immédiat des prints (utile pour SLURM)
os.environ.setdefault("PYTHONUNBUFFERED", "1")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.teacher import TeacherModel
from src.losses.gradient_matching import DepthAnythingLoss
from src.data.datasets import SyntheticDepthDataset
from src.data.transforms import TrainTransform, EvalTransform
from src.training.trainer import Trainer
from src.utils.helpers import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 3 — Entraînement Teacher sur données synthétiques"
    )

    # Données
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Répertoire du dataset synthétique (ex: datasets/synthetic/hypersim)")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Proportion de données pour la validation (défaut: 10%%)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Nombre max d'échantillons (pour debug/overfitting test)")

    # Modèle
    parser.add_argument("--image_size", type=int, default=518,
                        help="Taille des images (multiple de 14)")
    parser.add_argument("--decoder_hidden_dim", type=int, default=256,
                        help="Dimension interne du décodeur DPT")

    # Entraînement
    parser.add_argument("--epochs", type=int, default=25,
                        help="Nombre d'epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size (4-8 pour Giant model)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--gradient_clip", type=float, default=1.0,
                        help="Max norm gradient clipping")

    # Loss
    parser.add_argument("--lambda_ssi", type=float, default=0.5,
                        help="Lambda pour L_ssi")
    parser.add_argument("--alpha_gm", type=float, default=0.5,
                        help="Poids de L_gm")
    parser.add_argument("--top_k_masking", type=float, default=0.1,
                        help="Top-K%% masking")

    # Reprise
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint pour reprise d'entraînement")

    # Sortie
    parser.add_argument("--output_dir", type=str, default="outputs/teacher",
                        help="Répertoire de sortie")
    parser.add_argument("--seed", type=int, default=42,
                        help="Graine aléatoire")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Workers DataLoader")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    print("=" * 60, flush=True)
    print("Phase 3 : Entraînement du Teacher sur données synthétiques", flush=True)
    print("=" * 60, flush=True)

    # ----- 1. Modèle Teacher (backbone frozen, decoder trainable) -----
    print("\n--- Initialisation du Teacher ---", flush=True)
    print("    Chargement DINOv2-Giant (~4.5 GB)... cela peut prendre 1-2 min", flush=True)
    teacher = TeacherModel(
        pretrained_weights=None,        # Pas de poids pré-entraînés pour le decoder
        image_size=args.image_size,
        decoder_hidden_dim=args.decoder_hidden_dim,
        freeze_all=False,               # Mode entraînement : decoder trainable
    )

    params = teacher.count_parameters()
    print(f"  Total      : {params['total_M']:.1f}M", flush=True)
    print(f"  Trainable  : {params['trainable_M']:.1f}M (decoder)", flush=True)
    print(f"  Frozen     : {params['frozen_M']:.1f}M (DINOv2-Giant backbone)", flush=True)

    # Transférer sur GPU
    print(f"\n--- Transfert du modèle sur {device} ---", flush=True)
    teacher = teacher.to(device)
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(device).total_memory / 1e9
        print(f"  VRAM utilisée : {alloc:.1f} / {total:.1f} GB", flush=True)

    # ----- 2. Loss -----
    criterion = DepthAnythingLoss(
        lambda_ssi=args.lambda_ssi,
        alpha_gm=args.alpha_gm,
        top_k_masking=args.top_k_masking,
    )

    # ----- 3. Dataset synthétique -----
    print("\n--- Chargement des données synthétiques ---")
    train_transform = TrainTransform(image_size=args.image_size)
    eval_transform = EvalTransform(image_size=args.image_size)

    full_dataset = SyntheticDepthDataset(
        root=args.dataset_dir,
        transform=train_transform,
        max_samples=args.max_samples,
    )

    # Split train/val
    n_val = max(1, int(len(full_dataset) * args.val_ratio))
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    print(f"  Dataset    : {args.dataset_dir}")
    print(f"  Total      : {len(full_dataset)} images")
    print(f"  Train      : {n_train} images")
    print(f"  Val        : {n_val} images")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(2, args.num_workers),
        pin_memory=True,
    )

    # ----- 4. Configuration -----
    train_config = {
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "gradient_clip_max_norm": args.gradient_clip,
        "checkpoint_interval": 2,
        "early_stopping_patience": 7,
        "log_interval": 20,
    }

    # ----- 5. Trainer -----
    trainer = Trainer(
        model=teacher,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=str(device),
        output_dir=args.output_dir,
    )

    # Reprise depuis checkpoint
    if args.resume:
        trainer.resume_from_checkpoint(args.resume)

    # ----- 6. Entraînement -----
    print("\n--- Début de l'entraînement Teacher ---")
    results = trainer.train()

    print("\n" + "=" * 60)
    print("✅ Entraînement Teacher terminé.")
    print(f"   Meilleur modèle : {args.output_dir}/checkpoints/best_model.pt")
    print(f"   → Prochaine étape : générer les pseudo-labels (Phase 4.1)")
    print(f"     python scripts/generate_pseudo_labels.py \\")
    print(f"       --teacher_weights {args.output_dir}/checkpoints/best_model.pt \\")
    print(f"       --images_dir datasets/real_unlabeled/sa1b/images")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{'='*60}", flush=True)
        print(f"ERREUR FATALE : {type(e).__name__}: {e}", flush=True)
        print(f"{'='*60}", flush=True)
        traceback.print_exc()
        sys.exit(1)
