![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F518230%2F267432f8548b69a3ea6e2a39ea05c780%2Fhackathonlogo.png?generation=1585094525382728&alt=media)



## Description 

L’utilisation de modèles de Machine Learning pour la reconnaissance d’images est un sujet avancé d’un point de vue recherche, et qui commence à avoir des applications très matures dans certaines industries.
Dans le cadre d’images satellites, les applications sont multiples: De la cartographie d’urgence à l’occupation des sols en passant par la détection de nuages…
Néanmoins, obtenir des modèles de ML et de Deep Learning performants requiert souvent une phase de labellisation préalable très importante. Celle ci est manuelle, coûteuse, prend du temps, et peut être effectuée dans des conditions moyennement éthiques. 

Pourtant, les satellites acquièrent chaque jour une quantité très importante de données. 

**Comment tirer parti de cette quantité (non labellisée) de données afin d’améliorer nos modèles d’apprentissage supervisés ?**

A partir d’un très faible nombre d’images labellisées (environ 25 par classes), l’objectif de ce hackathon sera d’obtenir le meilleur classifieur possible (type compétition Kaggle) en tirant parti d’un grand nombre d’images sans labels.

Le sujet proposé est très en vogue en recherche (voir “self-supervised learning”), et les pistes à explorer sont multiples (pseudo labellisation, transfert learning, clustering, semi-supervision, self-supervision). Celles-ci laissent libre cours à la créativité des groupes pour “inventer” une méthode pour se passer des labels.

## Compétition

Vous aurez à disposition un très petit nombre d'exemples labellisés ainsi qu'un nombre plus important de données sans labels. 

L'objectif est d'obtenir le score de classification multiclasse (les classes sont exclusives) le plus élevé sur un jeu de test de 20000 exemples

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F518230%2F12fd4c83861a4cf356172f352dc75839%2Feurosat_1.png?generation=1585094363451532&alt=media)


## Sentinel2 / Eurosat

https://sentinel.esa.int/web/sentinel/missions/sentinel-2
https://www.statistiques.developpement-durable.gouv.fr/corine-land-cover-0

Les images satellites fournies par Sentinel 2 sont fournies de façon libre & gratuite dans le cadre du programme Copernicus de l'ESA.

Les satellites Sentinel 2 ont un pouvoir de revisite très important (une même zone est imagée en moyenne tous les 5 jours). Ce sont des instruments multispectraux acquérant des images allant jusqu'à 10m de résolution (1 pixel = 10m)

EUROSAT est un jeu de données d'occupation des sols basés sur dix classes d'occupation des sols et des images Sentinel 2. EuroSAT a été créé en 2017 par le German Research Center for Artificial Intelligence et est composé de 27000 imagettes (portion de 640m*640m soit 64x64px pour les bandes RGB). Les labels d'EuroSAT ont été revus manuellement par des opérateurs qualifiés

BigEarthNet quant à lui est un jeu de données d'occupation des sols basés sur 42 classes (Corine Land Cover officielle) et disposant de 600k imagettes (1200m*1200m soit 120x120px pour les bandes RGB)

## Acknowledgements

Les publications suivantes ont été utilisées lors de la construction des jeux de données

Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12(7), 2217-2226.

Sumbul, G., Charfuelan, M., Demir, B., & Markl, V. (2019, July). Bigearthnet: A large-scale benchmark archive for remote sensing image understanding. In IGARSS 2019-2019 IEEE International Geoscience and Remote Sensing Symposium (pp. 5901-5904). IEEE.

Sumbul, G., Kang, J., Kreuziger, T., Marcelino, F., Costa, H., Benevides, P., ... & Demir, B. (2020). BigEarthNet Deep Learning Models with A New Class-Nomenclature for Remote Sensing Image Understanding. arXiv preprint arXiv:2001.06372.

# Data

Les participants ont à leur disposition plusieurs jeux de données

### EuroSAT

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F518230%2F859ba5f1f4fbde2cd54d1008cca10ffb%2Feurosat_1.png?generation=1585094406047434&alt=media)


Ces jeux de données sont issus de la base de donnée Eurosat

- **eurosat_train.npz** : jeu de données labellisé de 500 images et labels associés à utiliser pour l'entraînement
- **eurosat_valid.npz** : jeu de données labellisé de 1500 images et labels associés à utiliser pour valider & tester son approche
- **eurosat_unsup.npz** : jeu de données non labellisé de 5000 images provnant de la même source que les images labellisées

La répartition des classes est équilibrée dans chacun des jeu à disposition

EuroSAT est une base de données disposant de 10 classes d'occupation des sols

    0: "AnnualCrop",
    1: "Forest",
    2: "HerbaceousVegetation",
    3: "Highway",
    4: "Industrial",
    5: "Pasture",
    6: "PermanentCrop",
    7: "Residential",
    8: "River",
    9: "SeaLake"

Ces classes sont exclusives.

**Il est interdit d'utiliser le jeu de données de validation pour l'entraînement**. Ce jeu n'est fourni qu'à titre d'exemple afin de faciliter les boucles de validation des modèles de Deep Learning et l'auto-évaluation.

### BigEarth Net

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F518230%2F3c5ffa7886f2dcc17bde227f5c7b3cef%2Fbigearthnet.png?generation=1585094441212219&alt=media)


Ce jeu de donnée est issu de la base de données BigEarthNet

- **bigearthnet_rgb.npz** : jeu de données de 95000 images non labellisées

### Propriétés des images

- BigEarthNet et Eurosat sont composées d'image 64x64 pixels en RGB

- Les images sont des "tuiles" d'images Sentinel2 (2A & 2B) traitées en "couleur naturelles" (True Color Image)

- Les images BigEarthNet et EuroSAT diffèrent par les traitements correctifs. Les images provenant de BigEarthNet sont dites de niveau "2" car les effets atmosphériques ont été corrigés. Les images provenant d'EuroSAT ne sont pas totalement corrigées, ce qui peut se traduire par un léger shift d'histogramme. 

**Jeu de test**

- **test.npz** jeu de test est composé de 20000 images de la base de données EUROSAT
- **sampleSubmission.csv** exemple de fichier csv à soumettre à Kaggle

## Format des données

Les données sont fournies au format .npz, c'est à dire "numpy compressed". Les images sont correctement formattées: Elles sont toutes de taille 64x64, disposent de trois canaux (RGB) et les valeurs sont contenues entre 0 et 255.

# Evaluation

## Métrique 

La métrique de cette compétition est le score de classification multiclasse, c'est à dire le total des prédictions correctes divisé par le nombre de prédictions effectuées.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F518230%2Fdd90b457f7bcce6487294b913df8f267%2FScreenshot%20from%202020-03-25%2000-27-37.png?generation=1585092415139404&alt=media)

Une prédiction est considérée correcte si et seulement si la classe prédite (au sens label textuel) est identique à la classe vraie.

## Format de soumission

Le format de soumission est le suivant:

- Format de fichier: .csv
- Colonnes: Id, Category

La colonne Id correspond à l'index dans le tableau numpy de test
La colonne Category correspond à la classe (label texte) prédite. Le mapping des classes est défini dans données.

Exemple:
```csv
Id,Cate**Eurosat**
gory
0,PermanentCrop
1,River
2,SeaLake
3,SeaLake
4,SeaLake
5,Pasture
6,Forest
7,SeaLake
8,AnnualCrop
9,Forest
10,Residential
```

- Aucun Id ne devrait être dupliqué
- Il ne peut y avoir qu'une seule réponse correcte pour chaque Id
- Des scripts d'exemples pour générer une soumission sont fournis dansle github d'exemples

