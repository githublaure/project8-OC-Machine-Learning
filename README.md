# Parcours Expert en ingénierie et science des données
# Projet 8 Parcours : Déployez un modèle dans le cloud 

# Fruits : extraction de features d'images avec PySpark (cloud)

Projet réalisé dans le cadre du parcours **Data Scientist** d'OpenClassrooms. Cette partie du projet met en place une chaîne de traitement *Big Data* dans le cloud : elle transforme des images de fruits en vecteurs de 50 valeurs, prêts à être utilisés pour une classification.

## Objectif

Pour chaque image de fruit stockée sur Amazon S3, le notebook :

1. calcule un vecteur de caractéristiques (*features*) avec **MobileNetV2** pré-entraîné sur ImageNet (transfer learning) ;
2. réduit ce vecteur de 1280 à **50 dimensions** avec une **PCA** (Spark ML) ;
3. sauvegarde le résultat sur S3 en **Parquet** et en **CSV**.

Le notebook ne classe pas les images : il prépare les données à grande échelle, en parallélisant le calcul sur un cluster Spark.


## Architecture

```
Images .jpg (S3)
      │  spark.read.format("binaryFile")
      ▼
DataFrame Spark (path, content, label)
      │  pandas UDF + MobileNetV2 (poids diffusés par broadcast)
      ▼
Features (1280 valeurs par image)
      │  conversion en VectorUDT, puis PCA (k = 50)
      ▼
pcaFeatures (50 valeurs par image)
      │
      ├──► Parquet partitionné   (Output_cloud)
      ├──► Parquet consolidé     (Output_cloud_consolidated/parquet)
      └──► CSV consolidé         (Output_cloud_consolidated/csv)
```

## Données

Les données sont lues et écrites directement sur S3 (bucket `fruits-p8-oc`), comme si elles étaient stockées localement.

| Chemin S3 | Contenu |
| --- | --- |
| `data/Test_cloud` | Images `.jpg`, un sous-dossier par classe de fruit |
| `data/Output_cloud` | Résultat en Parquet partitionné |
| `data/Output_cloud_consolidated/parquet` | Résultat en un seul fichier Parquet |
| `data/Output_cloud_consolidated/csv` | Résultat en un seul fichier CSV |

Le *label* de chaque image est extrait du nom de son dossier parent.

## Prérequis

- Un cluster Spark avec kernel **PySpark** et accès à S3 via le connecteur `s3a://` (le notebook a été exécuté sur un cluster cloud, probablement AWS EMR).
- Un bucket S3 contenant les images, dans la région `eu-west-1`.
- Les packages suivants, installés à la création du cluster via l'étape de **bootstrap** :
  - `tensorflow` (Keras, MobileNetV2)
  - `pillow`
  - `numpy`, `pandas`
  - `pyarrow`
  - `s3fs`, `boto3`

## Utilisation

1. Créer le cluster Spark en installant les packages ci-dessus pendant le bootstrap.
2. Déposer les images dans `data/Test_cloud`, avec un sous-dossier par classe.
3. Ouvrir le notebook sur le cluster avec le kernel PySpark.
4. Adapter les chemins de la cellule *Définition des PATH* si votre bucket a un autre nom.
5. Exécuter les cellules dans l'ordre.

## Détail du traitement

- **Modèle.** MobileNetV2 est chargé avec les poids ImageNet, sans sa dernière couche de classification. Chaque image donne un vecteur de 1280 valeurs.
- **Distribution du modèle.** Les poids sont diffusés (`broadcast`) à tous les workers, qui reconstruisent le modèle localement.
- **Prétraitement.** Redimensionnement en 224×224, conversion en tableau, puis `preprocess_input` de MobileNetV2.
- **Featurisation.** Une pandas UDF de type itérateur charge le modèle une fois par worker et l'applique par lots. Les images sont réparties sur 24 partitions.
- **Réduction de dimension.** `PCA(k=50)` de Spark ML sur les vecteurs de features.
- **Sauvegarde.** Parquet partitionné, Parquet consolidé (`coalesce(1)`) et CSV consolidé (colonnes `path`, `label`, `pcaFeatures_string`).

## Validation

En fin de notebook, `s3fs` et pandas retrouvent les fichiers CSV et Parquet consolidés, les relisent et vérifient que chaque vecteur PCA contient bien **50 dimensions**.

## Pistes d'amélioration

- Mettre en cache (`.cache()` ou `.persist()`) le DataFrame de features avant la PCA et les écritures, pour éviter de réexécuter le réseau de neurones plusieurs fois.
- Ne conserver que les colonnes utiles (`path`, `label`, `content`) avant la featurisation.
- Afficher `pca_model.explainedVariance` pour justifier le choix de k = 50.
- Ajouter un centrage des données (`StandardScaler(withMean=True)`) avant la PCA, car la PCA de Spark ML ne centre pas les données par défaut.
- Supprimer les cellules en double (imports, première version de `find_valid_file`) dans la version finale du notebook.
