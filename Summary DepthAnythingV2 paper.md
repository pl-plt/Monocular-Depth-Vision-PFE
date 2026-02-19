Voici un résumé de chaque section du document "depthAnyhtingV2.pdf" :

**Abstract**

Depth Anything V2 vise à construire un modèle d'estimation de profondeur monoculaire puissant en utilisant trois pratiques clés : remplacer toutes les images réelles étiquetées par des images synthétiques, augmenter la capacité du modèle enseignant (teacher model), et former les modèles étudiants (student models) via un pont d'images réelles pseudo-étiquetées à grande échelle. Contrairement aux modèles basés sur Stable Diffusion, Depth Anything V2 est significativement plus efficace et plus précis. Le travail propose des modèles de différentes échelles (de 25M à 1,3B de paramètres) et un nouveau banc d'évaluation (benchmark) polyvalent, nommé DA-2K, pour la recherche future.

**1 Introduction**

L'estimation de profondeur monoculaire (MDE) est un domaine en pleine croissance pour des applications variées comme la reconstruction 3D et la conduite autonome. Les modèles MDE se divisent en deux groupes : les modèles discriminatifs (comme Depth Anything V1) et les modèles génératifs (comme Marigold basé sur Stable Diffusion). Marigold excelle dans les détails fins, tandis que Depth Anything V1 est plus robuste pour les scènes complexes et plus efficace. Depth Anything V2 a pour objectif de combiner toutes les forces souhaitables, y compris la robustesse aux scènes complexes (objets transparents et surfaces réfléchissantes), les détails fins, l'efficacité, et la transférabilité. L'approche part de Depth Anything V1, se concentrant sur les données plutôt que sur des techniques sophistiquées, et en revisitant la conception des données étiquetées.

**2 Revisiting the Labeled Data Design of Depth Anything V1 (Revisiter la conception des données étiquetées de Depth Anything V1)**

Cette section explore les inconvénients des images réelles étiquetées, notamment le bruit d'étiquette (erreurs dues aux capteurs de profondeur, à la correspondance stéréo ou à la SfM) et les détails ignorés (estimations grossières). Ces problèmes rendent les modèles entraînés, comme MiDaS et Depth Anything V1, vulnérables aux objets transparents et aux réflexions. Le travail propose d'utiliser des images synthétiques avec des étiquettes de profondeur très précises, qui capturent tous les détails fins (y compris les maillages minces et les objets transparents/réfléchissants) et peuvent être rapidement augmentées.

**3 Challenges in Using Synthetic Data (Défis liés à l'utilisation de données synthétiques)**

Les images synthétiques présentent deux limites : 1\) un écart de distribution (elles sont trop "propres" et "ordonnées" par rapport aux images réelles) et 2\) une couverture de scène limitée (échantillonnées à partir de types de scènes fixes prédéfinis). Un test a montré que seul le modèle DINOv2-G, très lourd (1.3B de paramètres), pouvait se transférer avec succès du synthétique au réel, mais il échoue encore dans certains cas rares non couverts par les données synthétiques (comme le ciel ou les humains). Une solution simple de combiner des images réelles et synthétiques s'est avérée nuisible à la prédiction de détails fins.

**4 Key Role of Large-Scale Unlabeled Real Images (Rôle clé des images réelles non étiquetées à grande échelle)**

La solution proposée est d'incorporer des images réelles non étiquetées. Le modèle le plus performant (basé sur DINOv2-G et entraîné sur des images synthétiques) est d'abord utilisé pour assigner des pseudo-étiquettes de profondeur aux images réelles non étiquetées. Ensuite, les nouveaux modèles sont entraînés uniquement avec ces images pseudo-étiquetées à grande échelle. Ce processus sert à 1\) **combler l'écart de domaine** (les modèles se familiarisent avec la distribution des données réelles), 2\) **améliorer la couverture des scènes** (les images réelles non étiquetées sont plus diverses et informatives), et 3\) **transférer les connaissances** du modèle enseignant le plus performant aux modèles étudiants plus petits, ce qui est plus sûr que la distillation au niveau des fonctionnalités.

**5 Depth Anything V2**

Cette section présente le pipeline global de Depth Anything V2 en trois étapes : 1\) entraîner un modèle enseignant fiable (DINOv2-G) uniquement sur des images synthétiques de haute qualité ; 2\) produire des pseudo-étiquettes de profondeur précises sur des images réelles non étiquetées à grande échelle ; 3\) entraîner les modèles étudiants (basés sur DINOv2 small, base, large et giant) sur les images réelles pseudo-étiquetées pour une généralisation robuste. Les modèles utilisent cinq jeux de données synthétiques (595K images) et huit jeux de données réelles pseudo-étiquetées (62M images). L'entraînement utilise une perte invariante à l'échelle et au décalage ($\\mathcal{L}*{ssi}$) et une perte de correspondance de gradient ($\\mathcal{L}*{gm}$), cette dernière étant bénéfique à la netteté de la profondeur avec des images synthétiques. Des modèles de profondeur métrique sont obtenus par *fine-tuning* avec des étiquettes de profondeur métrique.

**6 A New Evaluation Benchmark: DA-2K (Un nouveau banc d'évaluation : DA-2K)**

Le document critique les bancs d'évaluation existants pour leurs limitations : 1\) **bruit d'étiquette** (annotations incorrectes, ex: miroirs et structures minces sur NYU-D), 2\) **diversité limitée** (se concentrant sur des scènes spécifiques), et 3\) **faible résolution** (environ 500x500). Pour pallier cela, DA-2K est construit pour l'estimation de profondeur monoculaire relative, avec un objectif de fournir 1\) une relation de profondeur précise, 2\) une couverture de scènes étendue, et 3\) des images à haute résolution. L'annotation se fait sur des paires de pixels (déterminer quel pixel est le plus proche), en utilisant un pipeline basé sur l'accord entre quatre modèles experts et l'intervention d'annotateurs humains en cas de désaccord, en plus d'une sélection manuelle pour les paires difficiles. Le benchmark couvre huit scénarios différents (Indoor, Outdoor, Non-real, Transparent/Reflective, Adverse style, Aerial, Underwater, Object).

**7 Experiment (Expérience)**

* **7.1 Implementation details (Détails d'implémentation)**: Le modèle utilise un décodeur DPT et des encodeurs DINOv2. L'entraînement du modèle enseignant sur les images synthétiques (160K itérations) est suivi par l'entraînement des modèles étudiants sur les images pseudo-étiquetées (480K itérations).  
* **7.2 Zero-Shot Relative Depth Estimation (Estimation de profondeur relative Zero-Shot)**: Sur les bancs d'évaluation conventionnels (Table 2), V2 est comparable à V1 et supérieur à MiDaS, mais les métriques ne reflètent pas les améliorations de V2 sur les détails fins et la robustesse. Sur le banc d'évaluation DA-2K (Table 3), même le plus petit modèle V2 surpasse significativement les autres modèles communautaires (Marigold, Geowizard, DepthFM), atteignant jusqu'à 10,6 % de meilleure précision que Marigold pour le modèle le plus performant.  
* **7.3 Fine-tuned to Metric Depth Estimation (*Fine-tuning* pour l'estimation de profondeur métrique)**: L'encodeur V2, une fois transféré à la tâche d'estimation de profondeur métrique (en remplaçant l'encodeur MiDaS dans le pipeline ZoeDepth), obtient des améliorations significatives sur les jeux de données NYU-D et KITTI par rapport aux méthodes précédentes. Deux modèles de profondeur métrique sont aussi entraînés sur les jeux de données synthétiques Hypersim et Virtual KITTI pour les scènes intérieures et extérieures, respectivement.  
* **7.4 Ablation Study (Étude d'ablation)**: Les images réelles pseudo-étiquetées augmentent considérablement les performances par rapport à l'entraînement uniquement sur des images synthétiques (Table 5). Entraîner les modèles étudiants uniquement sur des images pseudo-étiquetées donne des résultats légèrement meilleurs pour les modèles plus petits. Une comparaison quantitative sur le jeu de données DIML montre que les pseudo-étiquettes produites par V2 sont de bien meilleure qualité que les étiquettes manuelles d'origine (Table 6).

**8 Related Work (Travaux connexes)**

La section met en contexte Depth Anything V2 par rapport aux travaux précédents sur l'estimation de profondeur monoculaire (MDE). Les travaux récents se concentrent sur la MDE relative zero-shot, certains utilisant des modèles génératifs comme Stable Diffusion, tandis que d'autres (comme MiDaS et Depth Anything V1) se concentrent sur l'augmentation des données. Depth Anything V2 se distingue en mettant l'accent sur les images synthétiques pour la précision de la profondeur, tout en utilisant des images réelles pseudo-étiquetées à grande échelle et une stratégie d'augmentation de la taille du modèle enseignant pour résoudre le problème de généralisation. Le travail utilise la distillation de connaissances au niveau de la prédiction via des pseudo-étiquettes sur des données non étiquetées, ce qui est jugé plus sûr et plus facile que la distillation au niveau des fonctionnalités, surtout avec un grand écart de taille entre l'enseignant et l'étudiant.

**9 Conclusion**

Depth Anything V2 est un modèle de fondation plus puissant pour la MDE, offrant des prédictions robustes et détaillées, une prise en charge de tailles de modèles variées (25M à 1.3B) et une bonne capacité de *fine-tuning*. Le travail souligne l'importance de remplacer les images réelles étiquetées par des images synthétiques pour la précision. De plus, Depth Anything V2 introduit le banc d'évaluation DA-2K, qui couvre des scènes diverses et des images à haute résolution avec des étiquettes de profondeur éparses précises.

**A Sources of Training Data (Sources des données d'entraînement)**

Cette section détaille les sources de données utilisées : cinq jeux de données synthétiques précises (595K images, Table 7\) pour l'étiquetage, remplaçant les jeux de données réelles étiquetées de V1. Huit jeux de données réelles publiques à grande échelle (62M images, Table 7\) sont utilisées pour générer des pseudo-étiquettes, afin d'atténuer l'écart de distribution et la diversité limitée des images synthétiques.

**B Experiments (Expériences)**

Contient divers résultats d'expériences, y compris le *fine-tuning* vers la segmentation sémantique (Table 8), l'effet individuel des jeux de données synthétiques (Table 9\) et non étiquetées (Table 10), la nécessité d'images non étiquetées à grande échelle (Table 11), la performance sur les surfaces transparentes ou réfléchissantes (Table 12), la comparaison des encodeurs pré-entraînés (Table 13), l'effet de la perte de correspondance de gradient (Figure 10), et le suréchantillonnage de la résolution au moment du test (Figure 11). Il est démontré que les images réelles étiquetées nuisent aux prédictions détaillées (Figure 12).

**C DA-2K Evaluation Benchmark (Banc d'évaluation DA-2K)**

La section fournit des détails supplémentaires sur DA-2K, notamment la précision par scénario (Table 14), et une comparaison avec le jeu de données DIW. DA-2K est jugé plus précis, mieux organisé (par scénario), plus diversifié (incluant des images non réelles) et de plus haute résolution que DIW. Les mots-clés utilisés pour collecter les images par scénario sont listés (Table 15).

**D Limitations**

La principale limitation actuelle est la lourde charge de calcul due à l'utilisation de 62M d'images non étiquetées. De futurs travaux viseront à utiliser ces données plus efficacement et à collecter des images synthétiques plus diverses pour un meilleur pseudo-étiquetage.

Sources :

* [depthAnyhtingV2.pdf](https://drive.google.com/open?id=173EmpFXKlDbNWm4O7B7XVK_KtjsuHDNi)