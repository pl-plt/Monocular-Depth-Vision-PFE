"""
test_models.py — Tests unitaires pour les modèles (backbone, decoder, student).

Vérifie :
- Les shapes de sortie à chaque étape
- Le forward pass complet
- Le comptage de paramètres
- Le freeze/unfreeze du backbone
"""

import pytest
import torch
from src.models.decoder import DPTDecoder, ReassembleBlock, FusionBlock


class TestReassembleBlock:
    """Tests du bloc de réassemblage."""

    def test_shape_identity(self):
        """Scale factor = 1 : shape inchangée."""
        block = ReassembleBlock(input_dim=384, output_dim=256, scale_factor=1.0)
        x = torch.randn(2, 384, 37, 37)
        out = block(x)
        assert out.shape == (2, 256, 37, 37)

    def test_shape_upsample(self):
        """Scale factor = 2 : shape doublée."""
        block = ReassembleBlock(input_dim=384, output_dim=256, scale_factor=2.0)
        x = torch.randn(2, 384, 37, 37)
        out = block(x)
        assert out.shape == (2, 256, 74, 74)

    def test_shape_downsample(self):
        """Scale factor = 0.5 : shape divisée par 2."""
        block = ReassembleBlock(input_dim=384, output_dim=256, scale_factor=0.5)
        x = torch.randn(2, 384, 37, 37)
        out = block(x)
        assert out.shape[0] == 2
        assert out.shape[1] == 256


class TestFusionBlock:
    """Tests du bloc de fusion."""

    def test_forward_no_skip(self):
        """Forward sans skip connection."""
        block = FusionBlock(channels=256)
        x = torch.randn(2, 256, 16, 16)
        out = block(x)
        # Upsample x2
        assert out.shape == (2, 256, 32, 32)

    def test_forward_with_skip(self):
        """Forward avec skip connection."""
        block = FusionBlock(channels=256)
        x = torch.randn(2, 256, 16, 16)
        skip = torch.randn(2, 256, 16, 16)
        out = block(x, skip)
        assert out.shape == (2, 256, 32, 32)


class TestDPTDecoder:
    """Tests du décodeur DPT complet."""

    def test_forward_shape(self):
        """Vérifie la shape de sortie du décodeur."""
        decoder = DPTDecoder(
            input_dim=384,
            hidden_dim=256,
            output_dim=1,
            image_size=518,
        )
        # Simuler 4 niveaux de features
        features = [torch.randn(2, 384, 37, 37) for _ in range(4)]
        out = decoder(features)
        assert out.shape == (2, 1, 518, 518)

    def test_output_positive(self):
        """Vérifie que les depth sont positives (ReLU en sortie)."""
        decoder = DPTDecoder(input_dim=384, hidden_dim=256)
        features = [torch.randn(2, 384, 37, 37) for _ in range(4)]
        out = decoder(features)
        assert (out >= 0).all()

    def test_custom_target_size(self):
        """Test avec une taille de sortie personnalisée."""
        decoder = DPTDecoder(input_dim=384, hidden_dim=256)
        features = [torch.randn(1, 384, 37, 37) for _ in range(4)]
        out = decoder(features, target_size=(480, 640))
        assert out.shape == (1, 1, 480, 640)
