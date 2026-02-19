# Depth Anything V2 — PFE (Projet de Fin d'Études)

## Reproduction partielle de Depth Anything V2 avec ViT-Small

**Objectif :** Entraîner un modèle Student ViT-Small d'estimation de profondeur monoculaire via distillation à partir d'un Teacher DINOv2-Giant, en suivant la méthodologie de Depth Anything V2.

**Hardware cible :** NVIDIA H100  
**Durée :** 24 semaines (6 mois)

---

## Structure du projet

```
PFE/
├── configs/                    # Fichiers de configuration YAML
│   ├── data_config.yaml        # Configuration datasets & preprocessing
│   ├── train_config.yaml       # Hyperparamètres d'entraînement
│   └── eval_config.yaml        # Configuration évaluation & benchmarks
│
├── docs/                       # Documentation du projet
│   ├── Roadmap Projet Depth Anything V2.md
│   ├── Summary DepthAnythingV2 paper.md
│   ├── architecture.md         # Description architecture Teacher-Student
│   └── data_manifest.json      # Registre des datasets utilisés
│
├── src/                        # Code source principal
│   ├── models/                 # Architectures des modèles
│   │   ├── backbone.py         # Wrapper DINOv2 (extraction features multi-échelle)
│   │   ├── decoder.py          # Décodeur DPT (fusion features + upsampling)
│   │   ├── teacher.py          # Modèle Teacher (DINOv2-Giant + DPT, figé)
│   │   └── student.py          # Modèle Student (ViT-Small + DPT, entraînable)
│   │
│   ├── losses/                 # Fonctions de perte
│   │   ├── scale_invariant.py  # Scale-and-shift invariant loss (L_ssi)
│   │   └── gradient_matching.py# Gradient matching loss (L_gm)
│   │
│   ├── data/                   # Gestion des données
│   │   ├── datasets.py         # Classes Dataset PyTorch (synthétiques, réelles, pseudo-labels)
│   │   ├── transforms.py       # Augmentations et preprocessing
│   │   ├── download.py         # Scripts de téléchargement des datasets
│   │   └── preprocessing.py    # Nettoyage, filtrage, validation des données
│   │
│   ├── training/               # Logique d'entraînement
│   │   ├── trainer.py          # Boucle d'entraînement principale
│   │   ├── distillation.py     # Pipeline de distillation Teacher → Student
│   │   └── pseudo_labels.py    # Génération des pseudo-labels via le Teacher
│   │
│   ├── evaluation/             # Évaluation et analyse
│   │   ├── metrics.py          # Métriques : AbsRel, RMSE, δ1, δ2, δ3
│   │   ├── benchmark.py        # Évaluation sur NYU-Depth V2, KITTI
│   │   └── visualization.py    # Visualisation depth maps, comparaisons
│   │
│   └── utils/                  # Utilitaires
│       ├── logging_utils.py    # Intégration W&B / TensorBoard
│       ├── checkpoint.py       # Sauvegarde/chargement des checkpoints
│       └── helpers.py          # Fonctions utilitaires générales
│
├── scripts/                    # Points d'entrée (scripts exécutables)
│   ├── run_inference.py        # Phase 0 : inférence baseline avec modèle officiel
│   ├── extract_features.py     # Phase 1 : exploration features DINOv2
│   ├── download_data.py        # Phase 2 : téléchargement des datasets
│   ├── generate_pseudo_labels.py # Phase 4.1 : génération pseudo-labels
│   ├── train.py                # Phase 4.2-4.3 : entraînement Student
│   └── evaluate.py             # Phase 5 : évaluation sur benchmarks
│
├── notebooks/                  # Notebooks Jupyter d'exploration
│   ├── 01_baseline_inference.ipynb
│   ├── 02_dinov2_exploration.ipynb
│   ├── 03_data_exploration.ipynb
│   ├── 04_training_monitoring.ipynb
│   └── 05_evaluation_analysis.ipynb
│
├── tests/                      # Tests unitaires
│   ├── test_models.py
│   ├── test_losses.py
│   ├── test_data.py
│   └── test_metrics.py
│
├── outputs/                    # Résultats (gitignored)
│   ├── checkpoints/            # Poids sauvegardés
│   ├── logs/                   # Logs d'entraînement
│   ├── visualizations/         # Images générées
│   └── pseudo_labels/          # Pseudo-labels générés par le Teacher
│
├── datasets/                   # Données (gitignored)
│   ├── synthetic/              # Images synthétiques (Hypersim, etc.)
│   └── real_unlabeled/         # Images réelles non étiquetées (SA-1B, etc.)
│
├── requirements.txt            # Dépendances Python
├── .gitignore
└── README.md
```

---

## Installation

```bash
# Cloner le repo
git clone <repo-url>
cd PFE

# Créer environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou : venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt

# Vérifier GPU
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## Phases du projet

| Phase | Semaines | Description |
|-------|----------|-------------|
| 0 | S1 | Baseline & validation setup |
| 1 | S2-3 | Cadrage théorique & architecture |
| 2 | S4-9 | Data engineering |
| 3 | S10-14 | Implémentation architecture |
| 4 | S15-24 | Distillation & entraînement |
| 5 | S25-26 | Évaluation & analyse |

## Références

- [Depth Anything V2 (GitHub)](https://github.com/DepthAnything/Depth-Anything-V2)
- [DINOv2 (Facebook Research)](https://github.com/facebookresearch/dinov2)
- [Papier Depth Anything V2](https://arxiv.org/abs/2406.09414)
