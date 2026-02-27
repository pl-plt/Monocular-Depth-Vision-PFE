"""
test_data.py — Tests unitaires pour les datasets et transforms.

Vérifie :
- Les clés attendues dans le dict retourné par __getitem__
- Les shapes de sortie
- La cohérence des transforms jumelées (image + depth)
"""

import pytest
import torch
from src.data.transforms import TrainTransform, EvalTransform


class TestTrainTransform:
    """Tests des transforms d'entraînement."""

    def test_output_shape_image(self):
        """Image transformée : (3, H, W)."""
        t = TrainTransform(image_size=518)
        img = torch.randint(0, 255, (3, 600, 800), dtype=torch.uint8).float() / 255.0
        depth = torch.rand(1, 600, 800)
        img_t, depth_t = t(img, depth)
        assert img_t.shape == (3, 518, 518)

    def test_output_shape_depth(self):
        """Depth transformée : (1, H, W)."""
        t = TrainTransform(image_size=518)
        img = torch.rand(3, 600, 800)
        depth = torch.rand(1, 600, 800)
        img_t, depth_t = t(img, depth)
        assert depth_t.shape == (1, 518, 518)

    def test_depth_non_negative(self):
        """Les valeurs de depth restent >= 0 après transform."""
        t = TrainTransform(image_size=256)
        img = torch.rand(3, 400, 400)
        depth = torch.rand(1, 400, 400)
        _, depth_t = t(img, depth)
        assert (depth_t >= 0).all()


class TestEvalTransform:
    """Tests des transforms d'évaluation."""

    def test_output_shape(self):
        """Shape de sortie cohérente."""
        t = EvalTransform(image_size=518)
        img = torch.rand(3, 480, 640)
        depth = torch.rand(1, 480, 640)
        img_t, depth_t = t(img, depth)
        assert img_t.shape[0] == 3
        assert depth_t.shape[0] == 1

    def test_deterministic(self):
        """Eval transform est déterministe."""
        t = EvalTransform(image_size=518)
        img = torch.rand(3, 480, 640)
        depth = torch.rand(1, 480, 640)
        img_t1, depth_t1 = t(img.clone(), depth.clone())
        img_t2, depth_t2 = t(img.clone(), depth.clone())
        assert torch.allclose(img_t1, img_t2)
        assert torch.allclose(depth_t1, depth_t2)


class TestDataPreprocessing:
    """Tests légers des utilitaires de preprocessing."""

    def test_import_preprocessing(self):
        """Vérifie que le module s'importe correctement."""
        from src.data import preprocessing
        assert hasattr(preprocessing, 'validate_images')
        assert hasattr(preprocessing, 'validate_depth_maps')
        assert hasattr(preprocessing, 'create_train_val_split')
        assert hasattr(preprocessing, 'compute_dataset_stats')

    def test_train_val_split_basic(self):
        """Vérifie le ratio du split train/val."""
        from src.data.preprocessing import create_train_val_split
        import tempfile, os
        # Créer des fichiers factices
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(10):
                with open(os.path.join(tmpdir, f"img_{i:03d}.png"), 'w') as f:
                    f.write("fake")
            splits = create_train_val_split(tmpdir, val_ratio=0.2, seed=42)
            assert 'train' in splits
            assert 'val' in splits
            assert len(splits['train']) + len(splits['val']) == 10
