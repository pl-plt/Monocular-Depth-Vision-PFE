# Architecture Depth Anything V2 — Documentation technique

## Vue d'ensemble

Depth Anything V2 utilise une architecture **Teacher-Student** pour l'estimation
de profondeur monoculaire (Monocular Depth Estimation, MDE).

```
┌─────────────────────────────────────────────────────────┐
│                    PIPELINE GLOBAL                       │
│                                                          │
│  Étape 1 : Entraîner le Teacher sur images SYNTHÉTIQUES  │
│  ┌──────────────┐    ┌──────────┐    ┌──────────────┐   │
│  │ Images synth. │───▶│ Teacher  │───▶│ Depth GT     │   │
│  │ (Hypersim)    │    │ DINOv2-G │    │ (supervisé)  │   │
│  └──────────────┘    └──────────┘    └──────────────┘   │
│                                                          │
│  Étape 2 : Générer des pseudo-labels sur images RÉELLES  │
│  ┌──────────────┐    ┌──────────┐    ┌──────────────┐   │
│  │ Images réel.  │───▶│ Teacher  │───▶│ Pseudo-labels│   │
│  │ (SA-1B)       │    │ (figé)   │    │ (depth maps) │   │
│  └──────────────┘    └──────────┘    └──────────────┘   │
│                                                          │
│  Étape 3 : Entraîner le Student sur les pseudo-labels    │
│  ┌──────────────┐    ┌──────────┐    ┌──────────────┐   │
│  │ Images réel.  │───▶│ Student  │───▶│ Loss vs      │   │
│  │ + pseudo-lab. │    │ ViT-Small│    │ pseudo-labels│   │
│  └──────────────┘    └──────────┘    └──────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Architecture du modèle

### Encoder : DINOv2

DINOv2 est un **Vision Transformer (ViT)** pré-entraîné par Meta (Facebook Research)
via apprentissage auto-supervisé (self-supervised learning).

```
Image [B, 3, 518, 518]
    │
    ▼  Patch Embedding (14×14 patches)
Tokens [B, 1369, embed_dim]     (518/14 = 37 → 37×37 = 1369 tokens)
    │
    ▼  Transformer Blocks (× N layers)
    │
    ├── Layer k₁ → Features niveau 1 [B, embed_dim, 37, 37]
    ├── Layer k₂ → Features niveau 2 [B, embed_dim, 37, 37]
    ├── Layer k₃ → Features niveau 3 [B, embed_dim, 37, 37]
    └── Layer k₄ → Features niveau 4 [B, embed_dim, 37, 37]
```

| Variante | Layers | embed_dim | Params | Rôle |
|----------|--------|-----------|--------|------|
| ViT-Small | 12 | 384 | 25M | **Student** |
| ViT-Base | 12 | 768 | 86M | Student (opt.) |
| ViT-Large | 24 | 1024 | 300M | Student (opt.) |
| ViT-Giant | 40 | 1536 | 1.1B | **Teacher** |

### Decoder : DPT (Dense Prediction Transformer)

Le décodeur DPT fusionne les 4 niveaux de features et les upsample progressivement.

```
Features multi-échelle (4 niveaux, [B, C, 37, 37] chacun)
    │
    ▼  Reassemble Blocks (projection 1×1 + resize)
    ├── Niveau 1 → [B, 256, 148, 148]  (×4)
    ├── Niveau 2 → [B, 256, 74, 74]    (×2)
    ├── Niveau 3 → [B, 256, 37, 37]    (×1)
    └── Niveau 4 → [B, 256, 18, 18]    (×0.5)
    │
    ▼  Fusion Blocks (bottom-up progressive)
    │   Conv3×3 + BN + ReLU + Skip connection + Upsample ×2
    │
    ▼  Head (Conv3×3 → Conv1×1 → ReLU)
    │
Depth Map [B, 1, 518, 518]
```

## Fonctions de perte

### Scale-and-Shift Invariant Loss ($\mathcal{L}_{ssi}$)

$$\mathcal{L}_{ssi} = \sqrt{\frac{1}{n}\sum_{i}(d_i - d_i^*)^2 - \frac{\lambda}{n^2}\left(\sum_{i}(d_i - d_i^*)\right)^2}$$

- $d_i = \log(\text{pred}_i)$, $d_i^* = \log(\text{gt}_i)$
- $\lambda = 0.5$ (recommandé)
- Invariante à l'échelle et au décalage global

### Gradient Matching Loss ($\mathcal{L}_{gm}$)

$$\mathcal{L}_{gm} = \frac{1}{n}\sum_{i}||\nabla d_i - \nabla d_i^*||_1$$

- Encourage la netteté des bords
- Utilise des filtres de Sobel pour les gradients
- Bénéfique avec les images synthétiques

### Loss combinée

$$\mathcal{L} = \mathcal{L}_{ssi} + \alpha \cdot \mathcal{L}_{gm}$$

Avec masquage top-10% : les 10% de pixels avec les erreurs les plus élevées sont ignorés.

## Fichiers source correspondants

| Composant | Fichier | Description |
|-----------|---------|-------------|
| Backbone DINOv2 | `src/models/backbone.py` | Extraction features multi-échelle |
| Décodeur DPT | `src/models/decoder.py` | Fusion + upsampling |
| Teacher | `src/models/teacher.py` | DINOv2-Giant + DPT (figé) |
| Student | `src/models/student.py` | DINOv2-Small + DPT (entraînable) |
| Loss L_ssi | `src/losses/scale_invariant.py` | Scale-invariant loss |
| Loss L_gm | `src/losses/gradient_matching.py` | Gradient matching loss |
| Distillation | `src/training/distillation.py` | Pipeline Teacher → Student |
