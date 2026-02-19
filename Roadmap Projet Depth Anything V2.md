# **ROADMAP: Projet Depth Anything V2**

## **Lien discussion :** 

https://claude.ai/share/45eb5773-4d5c-49ef-b4c6-ab507072fe7f

## **PFE \- Recréation d'un algorithme de vision par ordinateur**

**Contexte :** Reproduction partielle de Depth Anything V2 avec une carte NVIDIA H100  
 **Durée totale :** 24 semaines (6 mois)  
 **Prérequis :** Connaissances Python, PyTorch, bases ML/Deep Learning  
 **Objectif :** Entraîner un modèle ViT-Small fonctionnel sur un subset réaliste de données

---

## **PHASE 0 : Baseline et Validation Setup (Semaine 1\)**

### **Objectifs**

Prouver que votre infrastructure fonctionne et établir une référence de performance claire.

### **Tâches concrètes**

1. **Setup environnement H100**

   * Installer PyTorch 2.x avec CUDA 12+  
   * Vérifier disponibilité GPU : `nvidia-smi`, test torch.cuda.is\_available()  
   * Installer dépendances : `transformers`, `timm`, `opencv-python`, `pillow`  
2. **Télécharger les poids officiels**

   * Récupérer Depth-Anything-V2-Small depuis le repo GitHub officiel  
   * Charger le modèle pré-entraîné en mémoire  
   * Vérifier la taille : \~100 MB, 25M paramètres  
3. **Tester l'inférence**

   * Télécharger 50-100 images de test (NYU-Depth V2 ou KITTI)  
   * Exécuter la prédiction sur ces images  
   * Visualiser les depth maps générées (colormap viridis)  
   * Mesurer le temps d'inférence moyen par image  
4. **Calculer les métriques de référence**

   * Sur un subset de NYU-D test set (654 images)  
   * Métriques : AbsRel, RMSE, δ1, δ2, δ3  
   * **Exemple target :** AbsRel \< 0.05, δ1 \> 0.99 (valeurs du papier)

### **Livrables**

* ✅ Script Python fonctionnel d'inférence  
* ✅ Notebook avec visualisations (images \+ depth maps)  
* ✅ Tableau de métriques baseline (vos résultats vs papier)  
* ✅ Confirmation accès H100 \+ temps GPU disponible/semaine

### **Critères de succès**

* Modèle officiel tourne sur votre H100 sans erreur  
* Temps inférence \< 0.5 sec/image  
* Métriques à ±5% des valeurs publiées

### **Plan B si blocage**

* Si GPU inaccessible : travailler en local avec CPU sur 10 images (lent mais validable)  
* Si poids introuvables : utiliser MiDaS v3 comme baseline alternative

---

## **PHASE 1 : Cadrage Théorique et Architecture (Semaines 2-3)**

### **Objectifs**

Maîtriser les fondamentaux mathématiques et techniques avant de coder.

### **Tâches concrètes**

**Semaine 2 : Étude du papier**

1. **Lecture approfondie**

   * Section sur l'architecture (ViT encoder \+ DPT decoder)  
   * Focus sur les 3 contributions clés :  
     * Entraînement Teacher sur données synthétiques uniquement  
     * Teacher DINOv2-Giant (1.1B params) pour pseudo-labels  
     * Student distillation sur images réelles non-étiquetées  
2. **Comprendre les loss functions**

   * Scale-invariant loss : $\\mathcal{L}\_{ssi} \= \\sqrt{\\frac{1}{n}\\sum(d\_i \- d\_i^*)^2 \- \\frac{\\lambda}{n^2}(\\sum(d\_i \- d\_i^*))^2}$  
   * Gradient matching loss : $\\mathcal{L}\_{gm} \= \\frac{1}{n}\\sum||\\nabla d\_i \- \\nabla d\_i^\*||\_1$  
   * Stratégie top-10% loss masking (ignorer pixels avec erreurs extrêmes)  
3. **Schématiser le pipeline**

   * Dessiner le flow : Image → DINOv2 features → Decoder → Depth map  
   * Identifier les poids figés vs entraînables

**Semaine 3 : Exploration DINOv2 et ViT**

**Tester DINOv2 pré-entraîné**

 from transformers import AutoModel  
dinov2 \= AutoModel.from\_pretrained("facebook/dinov2-giant")  
\# Extraire features sur une image test

from transformers import AutoModel

   dinov2 \= AutoModel.from\_pretrained("facebook/dinov2-giant")

   \# Extraire features sur une image test

1. **Comprendre l'architecture ViT**

   * Patch embedding (16x16 patches)  
   * Multi-head attention layers  
   * Output : features multi-échelle (4 niveaux)  
2. **Étudier le decoder DPT**

   * Comment fusionner les features multi-échelle  
   * Upsampling progressif vers résolution finale

### **Livrables**

* ✅ Document de synthèse (5-10 pages) : architecture, losses, data strategy  
* ✅ Schéma annoté du pipeline complet (draw.io ou PowerPoint)  
* ✅ Script test extraction features DINOv2 sur 10 images

### **Critères de succès**

* Vous pouvez expliquer le pipeline à votre encadrant sans notes  
* Features DINOv2 extraites avec shape correcte (e.g., \[B, 1536, H/14, W/14\])

---

## **PHASE 2 : Data Engineering (Semaines 4-9) \- 6 SEMAINES**

### **Objectifs**

Constituer et préparer vos datasets d'entraînement avec une approche pragmatique.

### **Semaine 4-5 : Sélection et téléchargement**

**Données synthétiques (pour validation, pas training)**

* Dataset : **Hypersim** (indoor synthetic)  
* Volume : 50,000 images (vs 595k original)  
* Sélection : échantillonnage stratifié (diversité scènes)  
* Stockage nécessaire : \~50 GB

**Données réelles non-étiquetées**

* Dataset : **SA-1B** (subset Segment Anything) ou **LSUN**  
* Volume cible progressif :  
  * Phase 1 : 50,000 images (proto rapide)  
  * Phase 2 : 200,000 images (si Phase 1 OK)  
  * Phase 3 : 500,000 images (si temps disponible)  
* Critère sélection : résolution minimale 512x512, diversité indoor/outdoor  
* Stockage : \~200-500 GB

**Script de téléchargement**

\# Exemple structure  
datasets/  
├── synthetic/  
│   ├── hypersim/  
│   │   ├── images/  
│   │   └── depth/  
└── real\_unlabeled/  
    └── sa1b/  
        └── images/

### **Semaine 6-7 : Preprocessing et dataloaders**

1. **Nettoyage des données**

   * Supprimer images corrompues (try/except PIL.Image.open)  
   * Filtrer images trop petites (\< 512px)  
   * Vérifier cohérence depth maps synthétiques

**Pipeline de preprocessing**

 transforms \= Compose(\[  
    Resize((518, 518)),  
    RandomHorizontalFlip(p=0.5),  
    RandomCrop((480, 480)),  
    ColorJitter(brightness=0.1, contrast=0.1),  
    ToTensor(),  
    Normalize(mean=\[0.485, 0.456, 0.406\],   
             std=\[0.229, 0.224, 0.225\])  
\])

2.   
3. **Dataloader optimisé**

   * `num_workers=8` (ajuster selon CPU)  
   * `prefetch_factor=2`  
   * `pin_memory=True` pour GPU  
   * Tester vitesse chargement : target \> 500 images/sec

### **Semaine 8-9 : Validation et versioning**

1. **Créer splits train/val**

   * Train : 90% des données  
   * Val : 10% (pour monitoring overfitting)  
2. **Data versioning**

   * Utiliser DVC ou simple fichier `data_manifest.json`  
   * Documenter : source, date téléchargement, preprocessing appliqué  
3. **Sanity checks**

   * Visualiser 50 exemples aléatoires  
   * Vérifier distribution tailles, ratios aspect  
   * Plot histogramme valeurs pixels

### **Livrables**

* ✅ Dataset 50k synthétiques téléchargé et vérifié  
* ✅ Dataset 50k réelles (minimum), idéalement 200k  
* ✅ Scripts preprocessing \+ dataloaders PyTorch  
* ✅ Documentation data (README avec stats, exemples)  
* ✅ Benchmark vitesse chargement

### **Critères de succès**

* Dataloader charge 500+ images/sec sur H100  
* Aucune image corrompue dans les datasets  
* Splits train/val bien séparés

### **Plan B si blocage**

* Si téléchargement trop long : utiliser ImageNet-1k (déjà disponible)  
* Si stockage insuffisant : réduire à 30k synthétiques \+ 100k réelles

---

## **PHASE 3 : Implémentation Architecture (Semaines 10-14) \- 5 SEMAINES**

### **Objectifs**

Implémenter et valider l'architecture Teacher-Student avant l'entraînement full-scale.

### **Semaine 10-11 : Modèle Teacher (DINOv2)**

**Charger DINOv2-Giant**

 teacher \= torch.hub.load('facebookresearch/dinov2', 'dinov2\_vitg14')  
teacher.eval()  \# Mode inference uniquement  
teacher.requires\_grad\_(False)  \# Figer tous les poids

1.   
2. **Ajouter le decoder head**

   * Option 1 : Réutiliser le code officiel Depth Anything V2  
   * Option 2 : Implémenter DPT decoder (plus long)  
   * **Recommandation : Option 1**  
3. **Tester forward pass Teacher**

   * Input : batch \[4, 3, 518, 518\]  
   * Output attendu : \[4, 1, 518, 518\] (depth maps)  
   * Vérifier shapes à chaque couche

### **Semaine 12-13 : Modèle Student (ViT-Small)**

**Initialiser ViT-Small backbone**

 from timm import create\_model  
student\_backbone \= create\_model('vit\_small\_patch14\_dinov2',   
                                pretrained=True)

1.   
2. **Ajouter decoder identique au Teacher**

   * Réutiliser exactement la même architecture de decoder  
   * Initialiser poids aléatoirement (sauf backbone pré-entraîné)

**Implémenter les loss functions**

 def scale\_invariant\_loss(pred, target, lambda\_=0.5):  
    \# Implémenter formule du papier  
    \# \+ masque top-10% erreurs  
    pass

def gradient\_matching\_loss(pred, target):  
    \# Sobel filters \+ L1 distance  
    pass

3. 

### **Semaine 14 : Validation sur toy data**

1. **Overfitting test (sanity check critique)**

   * Prendre 10 images \+ depth maps synthétiques  
   * Entraîner Student pour overfitter parfaitement  
   * **Target :** Loss \< 0.01 après 100 epochs  
   * Si ça ne marche pas → bug dans loss ou architecture  
2. **Validation gradient flow**

   * Vérifier gradients propagent jusqu'au backbone  
   * Pas de NaN ou explosion de gradients  
   * Utiliser `torch.autograd.grad_check`  
3. **Benchmark vitesse forward/backward**

   * Mesurer temps par batch (batch size 8, 16, 32\)  
   * Target : \< 0.5 sec/batch pour batch\_size=16

### **Livrables**

* ✅ Modèle Teacher fonctionnel (inference uniquement)  
* ✅ Modèle Student complet avec decoder  
* ✅ Fonctions de loss implémentées et testées  
* ✅ Preuve d'overfitting sur toy dataset (loss curve)  
* ✅ Code repository GitHub propre avec README

### **Critères de succès**

* Overfitting test réussi (loss \< 0.01 sur 10 images)  
* Shapes correctes à chaque étape du forward pass  
* Pas d'erreurs CUDA out-of-memory avec batch\_size=16

### **Plan B si blocage**

* Si implémentation DPT trop complexe : utiliser simple CNN decoder (moins performant mais fonctionne)  
* Si problèmes mémoire : réduire résolution à 384x384

---

## **PHASE 4 : Distillation et Entraînement (Semaines 15-24) \- 10 SEMAINES**

### **PHASE 4.1 : Génération Pseudo-Labels (Semaines 15-16)**

**Objectif :** Utiliser le Teacher pour créer les labels sur données réelles.

**Script batch inference**

 \# Pseudo-code  
teacher.eval()  
with torch.no\_grad():  
    for batch in unlabeled\_dataloader:  
        depth\_maps \= teacher(batch)  
        save\_depth\_maps(depth\_maps, batch\_ids)

1.   
2. **Calculs de temps**

   * 50,000 images × 0.2 sec/image \= **2.8 heures**  
   * 200,000 images × 0.2 sec/image \= **11 heures**  
   * Prévoir × 1.5 pour I/O et overheads  
3. **Stockage des pseudo-labels**

   * Format : numpy arrays (.npy) ou images 16-bit (.png)  
   * Compression si besoin (np.savez\_compressed)

Organisation :  
 pseudo\_labels/├── batch\_0000/├── batch\_0001/└── ...

*   
4. **Quality check visuel**

   * Visualiser 100 exemples aléatoires  
   * Vérifier cohérence : pas de depth maps aberrantes  
   * Plot distribution valeurs de profondeur

**Livrables :**

* ✅ 50k pseudo-labels générées (minimum)  
* ✅ Script inference automatisé et testé  
* ✅ Rapport qualité avec exemples visuels

**Critères succès :**

* Génération complète en \< 24h de compute  
* Aucun fichier corrompu  
* Depth maps visuellement cohérentes

---

### **PHASE 4.2 : Entraînement Initial (Semaines 17-20)**

**Objectif :** Premier entraînement complet du Student sur 50k images.

**Configuration training**

 \# Hyperparamètres recommandés  
batch\_size \= 16  
learning\_rate \= 1e-4  
epochs \= 20  
optimizer \= AdamW(student.parameters(), lr=lr, weight\_decay=0.01)  
scheduler \= CosineAnnealingLR(optimizer, T\_max=epochs)

1.   
2. **Training loop**

   * Sauvegarder checkpoint toutes les 2 epochs  
   * Logger métriques : loss, learning rate, GPU memory  
   * Utiliser Weights & Biases ou TensorBoard  
3. **Monitoring**

   * Plot courbes loss (train \+ val) en temps réel  
   * Early stopping si val loss stagne \> 5 epochs  
   * Vérifier overfitting : écart train/val loss  
4. **Debugging si convergence**

   * Si loss stagne : réduire LR (/10)  
   * Si explosion : gradient clipping (max\_norm=1.0)  
   * Si underfitting : augmenter epochs ou réduire weight decay

**Livrables :**

* ✅ Modèle entraîné sur 50k images  
* ✅ Courbes de training sauvegardées  
* ✅ Best checkpoint sélectionné

**Critères succès :**

* Loss décroissante sur au moins 15 epochs  
* Val loss \< 0.15 (approximatif, à ajuster)  
* Pas de crash GPU

**Plan B :**

* Si 50k trop long : réduire à 20k pour proto ultra-rapide  
* Si mémoire GPU insuffisante : batch\_size=8 \+ gradient accumulation

---

### **PHASE 4.3 : Scale-up et Optimisation (Semaines 21-24)**

**Objectif :** Entraîner sur dataset complet (200k+) et optimiser performances.

1. **Entraînement 200k images**

   * Reprendre meilleur checkpoint de Phase 4.2  
   * Fine-tuner sur dataset étendu  
   * Epochs : 10-15 (déjà pré-entraîné sur 50k)  
2. **Hyperparameter tuning**

   * Tester 2-3 learning rates (5e-5, 1e-4, 5e-4)  
   * Ajuster weight decay (0.01, 0.05)  
   * Expérimenter avec data augmentation strength  
3. **Ablation studies (si temps disponible)**

   * Tester impact de $\\mathcal{L}*{gm}$ vs seulement $\\mathcal{L}*{ssi}$  
   * Comparer top-10% masking vs pas de masking  
   * Essayer différentes résolutions (384, 518, 640\)  
4. **Monitoring avancé**

   * Calculer métriques sur val set toutes les 2 epochs  
   * Comparer avec baseline (modèle officiel)  
   * Target : gap \< 20% vs modèle officiel sur NYU-D

**Livrables :**

* ✅ Modèle final entraîné sur 200k+ images  
* ✅ Rapport d'ablation (si fait)  
* ✅ Comparaison metrics vs baseline

**Critères succès :**

* AbsRel \< 0.08 sur NYU-D test (vs 0.053 officiel)  
* δ1 \> 0.95 (vs 0.992 officiel)  
* Temps entraînement total \< 100 heures GPU

**Plan B :**

* Si 200k impossible dans temps imparti : rester à 100k  
* Si performances décevantes : analyser failure cases et documenter

---

## **PHASE 5 : Évaluation et Analyse (Semaines 25-26)**

### **Objectifs**

Quantifier performances et comprendre limites de votre modèle.

### **Semaine 25 : Évaluation quantitative**

1. **Benchmarks standards**

   * **NYU-Depth V2** (indoor) : 654 images test  
   * **KITTI** (outdoor/driving) : 697 images test  
   * Calculer toutes les métriques :  
     * AbsRel, RMSE, log10  
     * δ1, δ2, δ3 (accuracy thresholds)  
2. **Comparaison multi-modèles**

| Modèle | AbsRel (NYU) | δ1 (NYU) | Params |
| ----- | ----- | ----- | ----- |
| DAv2-Small (officiel) | 0.053 | 0.992 | 25M |
| Votre modèle (50k) | ? | ? | 25M |
| Votre modèle (200k) | ? | ? | 25M |

3.   
   **Analyse statistique**

   * Calculer intervalles de confiance (bootstrap)  
   * Identifier catégories d'images problématiques  
   * Breakdown par type de scène (indoor, outdoor, night, etc.)

### **Semaine 26 : Analyse qualitative**

1. **Visualisations**

   * Créer grille comparatives : Image | Ground Truth | Votre prédiction | DAv2 officiel  
   * Identifier 20 best cases et 20 worst cases  
   * Analyser patterns d'échecs  
2. **Failure mode analysis**

   * Quelles scènes posent problème ? (reflections, transparence, objets très fins)  
   * Erreurs liées aux données d'entraînement ?  
   * Différences indoor vs outdoor ?  
3. **Documentation**

   * Rédiger section "Résultats" du rapport  
   * Créer présentation avec visualisations clés  
   * Documenter différence performance vs papier original

### **Livrables**

* ✅ Tableau complet de métriques (tous benchmarks)  
* ✅ Notebook analyse qualitative avec visualisations  
* ✅ Section résultats rapport final (10-15 pages)  
* ✅ Slides présentation soutenance

### **Critères de succès**

* Métriques calculées sur au moins 2 benchmarks  
* Gap vs modèle officiel expliqué et documenté  
* Failure cases analysés en profondeur

---

## **BONUS OPTIONNEL : Déploiement Android (NON PRIORITAIRE)**

**⚠️ NE FAIRE QUE SI :**

* Phases 1-5 terminées avec ≥3 semaines d'avance  
* Au moins 1 membre de l'équipe a expérience Android  
* Projet principal déjà présentable pour soutenance

### **Si vous décidez de le faire**

**Semaine 27-28 : Conversion modèle**

* Export PyTorch → ONNX  
* ONNX → TensorFlow Lite  
* Quantization INT8 avec calibration dataset

**Semaine 29-30 : Application Android**

* Setup Android Studio \+ CameraX  
* Intégration TFLite interpreter  
* UI basique : preview \+ depth overlay

**Critère d'abandon :** Si après 1 semaine vous n'avez pas un prototype qui tourne (même lent), **abandonnez** et concentrez-vous sur l'amélioration du modèle ou l'analyse.

---

## **GESTION DE PROJET ET RECOMMANDATIONS**

### **Répartition équipe (si 2-3 personnes)**

**Personne 1 : Data \+ Infrastructure**

* Phases 2 et 4.1  
* Gestion datasets, dataloaders, cloud storage

**Personne 2 : Architecture \+ Training**

* Phases 3 et 4.2-4.3  
* Implémentation modèle, optimisation entraînement

**Personne 3 : Évaluation \+ Documentation**

* Phase 5  
* Benchmarking, visualisations, rédaction rapport

**En parallèle (tous) :** Phases 0, 1 (lecture commune)

### **Checkpoints hebdomadaires**

**Chaque vendredi :**

* Réunion 30min : état d'avancement vs planning  
* Identification blockers  
* Ajustement sprint suivant si nécessaire

**Livrables intermédiaires :**

* Semaine 3 : Présentation architecture à l'encadrant  
* Semaine 9 : Revue datasets  
* Semaine 16 : Démo pseudo-labels  
* Semaine 20 : Modèle v1.0 fonctionnel  
* Semaine 24 : Résultats préliminaires

### **Stratégie de mitigation des risques**

**Risque 1 : Accès GPU limité**

* Solution : Scripter jobs batch, lancer la nuit/weekend  
* Backup : Google Colab Pro (100h GPU/mois) ou AWS EC2 g4dn

**Risque 2 : Convergence impossible**

* Détection : Si après 10 epochs loss ne baisse pas  
* Solution : Revenir à overfitting test, debugger architecture  
* Plan C : Fine-tuner modèle officiel (toujours acceptable)

**Risque 3 : Manque de temps Phase 4**

* Décision go/no-go semaine 18  
* Si retard : rester à 50k images, approfondir analyse

### **Ressources techniques nécessaires**

**Hardware :**

* H100 : ≥50 heures compute (idéalement 100h)  
* Stockage : 1TB SSD/NVMe (datasets \+ checkpoints)  
* RAM : 64GB recommandé

**Software :**

* PyTorch ≥ 2.0  
* CUDA 12.x  
* Git \+ GitHub (versioning code)  
* Weights & Biases ou TensorBoard (monitoring)

---

## **CRITÈRES DE RÉUSSITE DU PFE**

### **Minima attendus (note ≥ 12/20)**

* ✅ Modèle Student implémenté et entraîné  
* ✅ Entraînement sur ≥50k images réussit  
* ✅ Évaluation sur au moins 1 benchmark (NYU ou KITTI)  
* ✅ Rapport complet avec méthodologie claire

### **Objectifs moyens (note ≥ 14/20)**

* ✅ Entraînement sur 200k images  
* ✅ Métriques \< 30% du modèle officiel  
* ✅ Évaluation sur 2 benchmarks  
* ✅ Analyse failure modes documentée

### **Excellence (note ≥ 16/20)**

* ✅ Métriques \< 20% du modèle officiel  
* ✅ Ablation studies avec insights originaux  
* ✅ Contributions au code open-source (PR sur repo officiel ?)  
* ✅ Démo interactive fonctionnelle

---

## **TIMELINE RÉCAPITULATIF**

Mois 1 (S1-4)   : Phase 0-1 (Baseline \+ Théorie)  
Mois 2-3 (S5-12) : Phase 2 (Data Engineering)  
Mois 3-4 (S13-16): Phase 3 \+ début Phase 4.1  
Mois 4-5 (S17-24): Phase 4.2-4.3 (Entraînement intensif)  
Mois 6 (S25-26)  : Phase 5 (Éval \+ Rapport)

**Marge de sécurité :** 2 semaines non planifiées pour imprévus, congés, ou approfondissements.

---

Cette roadmap est conçue pour être **réaliste, itérative et défaillible** : même si vous n'atteignez pas 200k images ou les performances optimales, vous aurez un projet complet et défendable. L'important est de documenter vos choix, vos obstacles, et vos apprentissages.

**Bon courage pour votre PFE \!** 🚀

