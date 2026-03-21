"""
test_data.py — Tests unitaires pour les datasets et transforms.

Vérifie :
- Les clés attendues dans le dict retourné par __getitem__
- Les shapes de sortie
- La cohérence des transforms jumelées (image + depth)
"""

import os
import tempfile
import pytest
import torch
import numpy as np
from PIL import Image as PILImage
from src.data.transforms import TrainTransform, EvalTransform


def _make_pil_pair(h=600, w=800):
    """Crée une paire (image PIL, depth PIL) pour les tests."""
    img = PILImage.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
    depth = PILImage.fromarray(np.random.rand(h, w).astype(np.float32))
    return img, depth


class TestTrainTransform:
    """Tests des transforms d'entraînement."""

    def test_output_shape_image(self):
        """Image transformée : (3, crop_size, crop_size)."""
        t = TrainTransform(image_size=518, crop_size=490)
        img, depth = _make_pil_pair()
        img_t, depth_t = t(img, depth)
        assert img_t.shape == (3, 490, 490)

    def test_output_shape_depth(self):
        """Depth transformée : (1, crop_size, crop_size)."""
        t = TrainTransform(image_size=518, crop_size=490)
        img, depth = _make_pil_pair()
        img_t, depth_t = t(img, depth)
        assert depth_t.shape == (1, 490, 490)

    def test_depth_non_negative(self):
        """Les valeurs de depth restent >= 0 après transform."""
        t = TrainTransform(image_size=256, crop_size=224)
        img, depth = _make_pil_pair(400, 400)
        _, depth_t = t(img, depth)
        assert (depth_t >= 0).all()


class TestEvalTransform:
    """Tests des transforms d'évaluation."""

    def test_output_shape(self):
        """Shape de sortie cohérente."""
        t = EvalTransform(image_size=518)
        img, depth = _make_pil_pair(480, 640)
        img_t, depth_t = t(img, depth)
        assert img_t.shape == (3, 518, 518)
        assert depth_t.shape == (1, 518, 518)

    def test_deterministic(self):
        """Eval transform est déterministe."""
        t = EvalTransform(image_size=518)
        img, depth = _make_pil_pair(480, 640)
        img_t1, depth_t1 = t(img, depth)
        img_t2, depth_t2 = t(img, depth)
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
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(10):
                with open(os.path.join(tmpdir, f"img_{i:03d}.png"), 'w') as f:
                    f.write("fake")
            train_files, val_files = create_train_val_split(tmpdir, val_ratio=0.2, seed=42)
            assert len(train_files) + len(val_files) == 10
            assert len(val_files) == 2  # 10 * 0.2 = 2
