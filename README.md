# Maple_Quebecois

Pipeline geoespacial para modelar la distribución potencial de **Acer saccharum** (sugar maple) y **Acer rubrum** (red maple) en Québec, integrando predictores bioclimáticos y covariables de coníferas.

## Contenido del repositorio

- `acer_grid_conifer_analysis.py`: script principal del flujo reproducible (validación de insumos, entrenamiento de modelos, predicción y salidas).
- `Supplementary_*.ipynb`: cuadernos de análisis complementarios y resultados suplementarios.
- `rasters_futuro.ipynb`: exploración/proyección de escenarios raster.
- `logos/`: recursos gráficos.

## ¿Qué hace el pipeline?

El script principal implementa un flujo de modelado en grilla que:

1. Repara/proyecta rasters ambientales a un CRS de análisis.
2. Construye covariables asociadas a coníferas (kernels y distancias).
3. Genera superficies de respuesta/presencia para sugar maple y red maple.
4. Entrena y evalúa dos familias de modelos por especie:
   - **Modelo ambiental** (solo variables ambientales).
   - **Modelo ambiental + coníferas**.
5. Exporta mapas de probabilidad, diferencias entre modelos y tablas de importancia de variables.

# Maple_Quebecois

Geospatial modelling pipeline for assessing the present-day and future suitability of sugar maple (*Acer saccharum*) and red maple (*Acer rubrum*) across Québec, Canada. The workflow integrates bioclimatic, edaphic and conifer-associated covariates to compare environment-only models with models that include ecological filters linked to the boreal conifer matrix.

---

# English

## Overview

`Maple_Quebecois` is a reproducible geospatial modelling pipeline developed to analyse the distribution and future climatic suitability of two maple species in Québec:

- Sugar maple (*Acer saccharum*)
- Red maple (*Acer rubrum*)

The pipeline uses grid-based species distribution modelling with XGBoost. It compares two modelling configurations:

1. Model A — Environment-only model  
   Uses climatic and edaphic predictors.

2. Model B — Environment + conifer-associated ecological filters  
   Uses the same environmental predictors plus covariates derived from the spatial distribution of three boreal conifer species:

   - Black spruce (*Picea mariana*)
   - Jack pine (*Pinus banksiana*)
   - Balsam fir (*Abies balsamea*)

The conifer-associated covariates are not interpreted as direct causal exclusion mechanisms. Instead, they are used as spatial indicators of the surrounding boreal forest matrix and associated ecological conditions that may constrain realised maple suitability.

---

## Repository contents

Maple_Quebecois/
│
├── acer_grid_conifer_analysis.py
│   Main reproducible pipeline for raster validation, covariate creation,
│   model training, spatial prediction and output generation.
│
├── Supplementary_*.ipynb
│   Supplementary notebooks containing extended analyses, model diagnostics,
│   figures and additional results.
│
├── rasters_futuro.ipynb
│   Notebook for inspecting and preparing future raster projections.
│
├── logos/
│   Graphic resources used in notebooks or documentation.
│
├── LICENSE
│   Repository licence.
│
└── README.md

---

## What the pipeline does

The main script implements a complete grid-based modelling workflow:

1. Validates and aligns environmental raster layers.
2. Reprojects environmental rasters to the analysis coordinate reference system.
3. Builds conifer-associated covariates:
   - Gaussian kernel density surfaces.
   - Distance-to-nearest-conifer-presence rasters.
4. Rasterises maple occurrence records to a common grid.
5. Builds species-specific training tables.
6. Trains and evaluates two XGBoost model configurations per species:
   - Environment-only model.
   - Environment + conifer model.
7. Exports present-day probability maps.
8. Exports model-difference rasters.
9. Computes feature-importance outputs and SHAP summaries.
10. Projects future suitability under multiple climate scenarios.
11. Summarises projected changes in suitable area and latitudinal shifts.

---

## Expected data structure

The pipeline expects a project directory with a structure similar to:

project_directory/
│
├── bioclim_data/
│   └── recortados_alineados/
│       Environmental raster layers.
│
├── data/
│   ├── db_sugar_maple.csv
│   ├── db_red_maple.csv
│   ├── data_final_forestry_2.csv
│   │
│   └── Politic_divition/
│       └── lpr_000b21a_e.shp
│
├── futuros_clip/
│   Future climate raster layers.
│
├── derived_rasters/
│   Automatically generated derived rasters.
│
├── model_inputs/
│   Automatically generated model input tables and intermediate files.
│
├── models/
│   Serialized trained models.
│
└── outputs/
    ├── sugar/
    └── red/

The names above reflect the current project structure. If the input file names or folder locations change, the corresponding paths should be updated in `acer_grid_conifer_analysis.py`.

---

## Main input files

| File or folder | Description |
|---|---|
| `bioclim_data/recortados_alineados/` | Present-day environmental raster layers. |
| `futuros_clip/` | Future climate raster layers used for projection. |
| `data/db_sugar_maple.csv` | Occurrence data for sugar maple. |
| `data/db_red_maple.csv` | Occurrence data for red maple. |
| `data/data_final_forestry_2.csv` | Auxiliary forestry data used to derive conifer-associated covariates. |
| `data/Politic_divition/lpr_000b21a_e.shp` | Spatial boundary layer used for Québec masking and visualisation. |

---

## Main outputs

For each species, the pipeline generates outputs under:

outputs/sugar/
outputs/red/

Typical outputs include:

| Output | Description |
|---|---|
| `training_table_*.csv` | Grid-level training table. |
| `model_comparison_*.csv` | Model-performance comparison. |
| `cv_metrics_*.csv` | Cross-validation metrics. |
| `present_*_probability.tif` | Present-day predicted suitability rasters. |
| `difference_*_env_minus_conifer.tif` | Difference rasters comparing Model A and Model B. |
| `feature_importance_*.csv` | Feature-importance summaries. |
| `shap_summary_*` | SHAP summary outputs. |
| `future_projection_metrics_*.csv` | Future suitability metrics across scenarios. |
| `future_predictions/` | Future suitability probability rasters. |
| `stats/` | Additional statistical outputs, when generated. |

---

## Requirements

Python 3.10 or later is recommended.

Main Python dependencies:

geopandas
rasterio
numpy
pandas
scipy
scikit-learn
xgboost
shap
matplotlib

Suggested installation:

pip install geopandas rasterio numpy pandas scipy scikit-learn xgboost shap matplotlib

On some systems, especially Windows, `geopandas` and `rasterio` may require GDAL and PROJ dependencies. If installation with `pip` fails, consider using `conda`:

conda install -c conda-forge geopandas rasterio numpy pandas scipy scikit-learn xgboost shap matplotlib

---

## Quick start

### 1. Validate inputs without training models

python acer_grid_conifer_analysis.py --project-dir /path/to/project --species both --check-only

### 2. Run the full analysis for both species

python acer_grid_conifer_analysis.py --project-dir /path/to/project --species both

### 3. Run the analysis for sugar maple only

python acer_grid_conifer_analysis.py --project-dir /path/to/project --species sugar

### 4. Run the analysis for red maple only

python acer_grid_conifer_analysis.py --project-dir /path/to/project --species red

---

## Command-line arguments

| Argument | Description |
|---|---|
| `--project-dir` | Base project directory containing inputs and output folders. |
| `--species` | Species to analyse: `sugar`, `red`, or `both`. |
| `--check-only` | Runs input validation without model training or prediction. |

---

## Modelling framework

The workflow compares two model configurations:

### Model A — Environment-only

This model estimates maple suitability using environmental predictors only, including climatic and edaphic variables.

### Model B — Environment + conifer-associated filters

This model extends Model A by adding conifer-associated covariates derived from black spruce, jack pine and balsam fir occurrence patterns. These covariates include:

- Kernel-density surfaces.
- Distance-to-nearest-presence rasters.

The purpose of Model B is to test whether the boreal conifer matrix improves the estimation of realised maple suitability, especially near the northern distributional limits.

---

## Future projections

Future suitability is projected under multiple climate-model, emission-scenario and time-period combinations. The pipeline produces two types of future projections:

1. Environment-only future projections  
   Model A is transferred to future environmental conditions.

2. Static-conifer sensitivity projections  
   Model B is transferred to future environmental conditions while conifer-associated covariates are held fixed at present-day values.

The static-conifer configuration should not be interpreted as a realised future forest distribution. It is a sensitivity analysis that asks how future maple suitability changes if the current conifer-associated forest structure remains spatially fixed.

---

## Reproducibility notes

The pipeline is designed to make the modelling workflow transparent and reproducible. However, users should verify:

- Raster alignment.
- Coordinate reference systems.
- NoData handling.
- Input variable names.
- Scenario naming conventions.
- Availability and licensing of third-party datasets.

Large raster files and derived outputs may not be included directly in the GitHub repository. When necessary, these should be archived separately in a data repository such as Zenodo.

---

## Data availability

The analysis code and reproducible workflow are provided in this repository. Large input rasters, derived rasters and selected model outputs may be archived separately through Zenodo or another public repository, depending on data redistribution permissions.

Some input data may originate from third-party sources and remain subject to their original access and licensing conditions.

---

## Licence

This repository is distributed under the licence specified in the `LICENSE` file.

---

# Español

## Descripción general

`Maple_Quebecois` es un pipeline geoespacial reproducible desarrollado para analizar la distribución actual y la idoneidad futura de dos especies de arce en Québec, Canadá:

- Arce de azúcar (*Acer saccharum*)
- Arce rojo (*Acer rubrum*)

El flujo de trabajo utiliza modelos de distribución de especies basados en una grilla común y entrenados con XGBoost. Se comparan dos configuraciones de modelado:

1. Modelo A — Modelo ambiental  
   Incluye únicamente predictores climáticos y edáficos.

2. Modelo B — Modelo ambiental + filtros ecológicos asociados a coníferas  
   Incluye los mismos predictores ambientales más covariables derivadas de la distribución espacial de tres especies de coníferas boreales:

   - Picea negra (*Picea mariana*)
   - Pino banksiano (*Pinus banksiana*)
   - Abeto balsámico (*Abies balsamea*)

Las covariables asociadas a coníferas no se interpretan como mecanismos causales directos de exclusión. Se usan como indicadores espaciales de la matriz boreal de coníferas y de las condiciones ecológicas asociadas que pueden limitar la idoneidad realizada de los arces.

---

## Contenido del repositorio

Maple_Quebecois/
│
├── acer_grid_conifer_analysis.py
│   Script principal del flujo reproducible para validación de rasters,
│   construcción de covariables, entrenamiento de modelos, predicción
│   espacial y generación de salidas.
│
├── Supplementary_*.ipynb
│   Cuadernos suplementarios con análisis extendidos, diagnósticos,
│   figuras y resultados adicionales.
│
├── rasters_futuro.ipynb
│   Cuaderno para inspección y preparación de rasters climáticos futuros.
│
├── logos/
│   Recursos gráficos utilizados en notebooks o documentación.
│
├── LICENSE
│   Licencia del repositorio.
│
└── README.md

---

## ¿Qué hace el pipeline?

El script principal implementa un flujo completo de modelado geoespacial en grilla:

1. Valida y alinea rasters ambientales.
2. Reproyecta rasters al sistema de coordenadas de análisis.
3. Construye covariables asociadas a coníferas:
   - Superficies de densidad tipo kernel.
   - Rasters de distancia a la presencia más cercana.
4. Rasteriza registros de presencia de arces en una grilla común.
5. Construye tablas de entrenamiento por especie.
6. Entrena y evalúa dos configuraciones de modelos XGBoost por especie:
   - Modelo ambiental.
   - Modelo ambiental + coníferas.
7. Exporta mapas de probabilidad actual.
8. Exporta rasters de diferencia entre modelos.
9. Calcula importancia de variables y resúmenes SHAP.
10. Proyecta la idoneidad futura bajo múltiples escenarios climáticos.
11. Resume cambios proyectados en área idónea y desplazamientos latitudinales.

---

## Estructura esperada de datos

El pipeline espera una estructura similar a:

directorio_del_proyecto/
│
├── bioclim_data/
│   └── recortados_alineados/
│       Rasters ambientales actuales.
│
├── data/
│   ├── db_sugar_maple.csv
│   ├── db_red_maple.csv
│   ├── data_final_forestry_2.csv
│   │
│   └── Politic_divition/
│       └── lpr_000b21a_e.shp
│
├── futuros_clip/
│   Rasters climáticos futuros.
│
├── derived_rasters/
│   Rasters derivados generados automáticamente.
│
├── model_inputs/
│   Tablas e insumos intermedios generados automáticamente.
│
├── models/
│   Modelos entrenados serializados.
│
└── outputs/
    ├── sugar/
    └── red/

Los nombres anteriores reflejan la estructura actual del proyecto. Si cambian los nombres o ubicaciones de los insumos, se deben actualizar las rutas correspondientes en `acer_grid_conifer_analysis.py`.

---

## Archivos principales de entrada

| Archivo o carpeta | Descripción |
|---|---|
| `bioclim_data/recortados_alineados/` | Rasters ambientales actuales. |
| `futuros_clip/` | Rasters climáticos futuros usados para proyección. |
| `data/db_sugar_maple.csv` | Datos de ocurrencia de arce de azúcar. |
| `data/db_red_maple.csv` | Datos de ocurrencia de arce rojo. |
| `data/data_final_forestry_2.csv` | Datos forestales auxiliares usados para construir covariables de coníferas. |
| `data/Politic_divition/lpr_000b21a_e.shp` | Capa espacial de límites utilizada para máscara y visualización de Québec. |

---

## Salidas principales

Para cada especie, las salidas se generan en:

outputs/sugar/
outputs/red/

Salidas típicas:

| Salida | Descripción |
|---|---|
| `training_table_*.csv` | Tabla de entrenamiento a nivel de grilla. |
| `model_comparison_*.csv` | Comparación de desempeño entre modelos. |
| `cv_metrics_*.csv` | Métricas de validación cruzada. |
| `present_*_probability.tif` | Rasters de idoneidad actual predicha. |
| `difference_*_env_minus_conifer.tif` | Rasters de diferencia entre Modelo A y Modelo B. |
| `feature_importance_*.csv` | Resúmenes de importancia de variables. |
| `shap_summary_*` | Salidas asociadas a SHAP. |
| `future_projection_metrics_*.csv` | Métricas de idoneidad futura por escenario. |
| `future_predictions/` | Rasters de probabilidad de idoneidad futura. |
| `stats/` | Salidas estadísticas adicionales, cuando se generan. |

---

## Requisitos

Se recomienda Python 3.10 o superior.

Dependencias principales:

geopandas
rasterio
numpy
pandas
scipy
scikit-learn
xgboost
shap
matplotlib

Instalación sugerida:

pip install geopandas rasterio numpy pandas scipy scikit-learn xgboost shap matplotlib

En algunos sistemas, especialmente Windows, `geopandas` y `rasterio` pueden requerir dependencias de GDAL y PROJ. Si la instalación con `pip` falla, se recomienda usar `conda`:

conda install -c conda-forge geopandas rasterio numpy pandas scipy scikit-learn xgboost shap matplotlib

---

## Uso rápido

### 1. Validar insumos sin entrenar modelos

python acer_grid_conifer_analysis.py --project-dir /ruta/al/proyecto --species both --check-only

### 2. Ejecutar análisis completo para ambas especies

python acer_grid_conifer_analysis.py --project-dir /ruta/al/proyecto --species both

### 3. Ejecutar análisis solo para arce de azúcar

python acer_grid_conifer_analysis.py --project-dir /ruta/al/proyecto --species sugar

### 4. Ejecutar análisis solo para arce rojo

python acer_grid_conifer_analysis.py --project-dir /ruta/al/proyecto --species red

---

## Argumentos de línea de comandos

| Argumento | Descripción |
|---|---|
| `--project-dir` | Directorio base del proyecto con insumos y carpetas de salida. |
| `--species` | Especie a analizar: `sugar`, `red` o `both`. |
| `--check-only` | Ejecuta validación de insumos sin entrenar modelos ni generar predicciones. |

---

## Marco de modelado

El flujo compara dos configuraciones:

### Modelo A — Ambiental

Estima la idoneidad de los arces usando solo predictores ambientales, incluyendo variables climáticas y edáficas.

### Modelo B — Ambiental + filtros asociados a coníferas

Extiende el Modelo A al incluir covariables asociadas a la presencia de picea negra, pino banksiano y abeto balsámico. Estas covariables incluyen:

- Superficies de densidad tipo kernel.
- Distancia a la presencia más cercana.

El objetivo del Modelo B es evaluar si la matriz boreal de coníferas mejora la estimación de la idoneidad realizada de los arces, especialmente cerca de sus límites norteños de distribución.

---

## Proyecciones futuras

La idoneidad futura se proyecta bajo múltiples combinaciones de modelos climáticos, escenarios de emisión y periodos temporales. El pipeline genera dos tipos de proyecciones:

1. Proyecciones futuras ambientales  
   El Modelo A se transfiere a condiciones ambientales futuras.

2. Proyecciones de sensibilidad con coníferas estáticas  
   El Modelo B se transfiere a condiciones ambientales futuras, manteniendo las covariables asociadas a coníferas fijas en sus valores actuales.

La configuración de coníferas estáticas no debe interpretarse como una distribución futura realizada del bosque. Es un análisis de sensibilidad que evalúa cómo cambia la idoneidad futura de los arces si la estructura espacial actual de coníferas permanece fija.

---

## Notas de reproducibilidad

El pipeline busca que el flujo de modelado sea transparente y reproducible. Sin embargo, los usuarios deben verificar:

- Alineación de rasters.
- Sistemas de coordenadas.
- Manejo de valores NoData.
- Nombres de variables de entrada.
- Nombres de escenarios futuros.
- Disponibilidad y licencias de datos de terceros.

Los archivos raster grandes y algunas salidas derivadas pueden no estar incluidos directamente en GitHub. Cuando sea necesario, estos archivos deben archivarse en un repositorio de datos como Zenodo.

---

## Disponibilidad de datos

El código de análisis y el flujo reproducible están disponibles en este repositorio. Los rasters de entrada de gran tamaño, rasters derivados y algunas salidas del modelo pueden archivarse por separado en Zenodo u otro repositorio público, dependiendo de las condiciones de redistribución de los datos originales.

Algunos insumos pueden provenir de fuentes de terceros y estar sujetos a sus condiciones originales de acceso y licencia.

---

## Licencia

Este repositorio se distribuye bajo la licencia indicada en el archivo `LICENSE`.

---

# Français

## Aperçu

`Maple_Quebecois` est un pipeline géospatial reproductible développé pour analyser la distribution actuelle et la pertinence écologique future de deux espèces d’érables au Québec, Canada :

- Érable à sucre (*Acer saccharum*)
- Érable rouge (*Acer rubrum*)

Le flux de travail utilise une modélisation de distribution d’espèces basée sur une grille commune et des modèles XGBoost. Deux configurations de modélisation sont comparées :

1. Modèle A — Modèle environnemental  
   Utilise uniquement des prédicteurs climatiques et édaphiques.

2. Modèle B — Modèle environnemental + filtres écologiques associés aux conifères  
   Utilise les mêmes prédicteurs environnementaux ainsi que des covariables dérivées de la distribution spatiale de trois espèces de conifères boréaux :

   - Épinette noire (*Picea mariana*)
   - Pin gris (*Pinus banksiana*)
   - Sapin baumier (*Abies balsamea*)

Les covariables associées aux conifères ne sont pas interprétées comme des mécanismes causaux directs d’exclusion. Elles sont utilisées comme indicateurs spatiaux de la matrice forestière boréale et des conditions écologiques associées pouvant limiter la pertinence écologique réalisée des érables.

---

## Contenu du dépôt

Maple_Quebecois/
│
├── acer_grid_conifer_analysis.py
│   Script principal du flux reproductible pour la validation des rasters,
│   la création des covariables, l’entraînement des modèles, la prédiction
│   spatiale et la génération des sorties.
│
├── Supplementary_*.ipynb
│   Carnets supplémentaires contenant des analyses étendues, des diagnostics,
│   des figures et des résultats additionnels.
│
├── rasters_futuro.ipynb
│   Carnet pour inspecter et préparer les projections raster futures.
│
├── logos/
│   Ressources graphiques utilisées dans les carnets ou la documentation.
│
├── LICENSE
│   Licence du dépôt.
│
└── README.md

---

## Fonctionnement du pipeline

Le script principal met en œuvre un flux complet de modélisation géospatiale sur grille :

1. Valide et aligne les couches raster environnementales.
2. Reprojette les rasters environnementaux vers le système de coordonnées d’analyse.
3. Construit des covariables associées aux conifères :
   - Surfaces de densité par noyau gaussien.
   - Rasters de distance à la présence de conifère la plus proche.
4. Rasterise les occurrences d’érables sur une grille commune.
5. Construit des tables d’entraînement spécifiques à chaque espèce.
6. Entraîne et évalue deux configurations de modèles XGBoost par espèce :
   - Modèle environnemental.
   - Modèle environnemental + conifères.
7. Exporte des cartes de probabilité actuelle.
8. Exporte des rasters de différence entre modèles.
9. Calcule l’importance des variables et les résumés SHAP.
10. Projette la pertinence écologique future sous plusieurs scénarios climatiques.
11. Résume les changements projetés d’aire favorable et les déplacements latitudinaux.

---

## Structure attendue des données

Le pipeline attend un répertoire de projet structuré de manière similaire à :

repertoire_du_projet/
│
├── bioclim_data/
│   └── recortados_alineados/
│       Couches raster environnementales actuelles.
│
├── data/
│   ├── db_sugar_maple.csv
│   ├── db_red_maple.csv
│   ├── data_final_forestry_2.csv
│   │
│   └── Politic_divition/
│       └── lpr_000b21a_e.shp
│
├── futuros_clip/
│   Couches climatiques futures.
│
├── derived_rasters/
│   Rasters dérivés générés automatiquement.
│
├── model_inputs/
│   Tables d’entrée et fichiers intermédiaires générés automatiquement.
│
├── models/
│   Modèles entraînés sérialisés.
│
└── outputs/
    ├── sugar/
    └── red/

Les noms ci-dessus reflètent la structure actuelle du projet. Si les noms ou les emplacements des fichiers changent, les chemins correspondants doivent être mis à jour dans `acer_grid_conifer_analysis.py`.

---

## Principaux fichiers d’entrée

| Fichier ou dossier | Description |
|---|---|
| `bioclim_data/recortados_alineados/` | Couches raster environnementales actuelles. |
| `futuros_clip/` | Couches climatiques futures utilisées pour la projection. |
| `data/db_sugar_maple.csv` | Données d’occurrence de l’érable à sucre. |
| `data/db_red_maple.csv` | Données d’occurrence de l’érable rouge. |
| `data/data_final_forestry_2.csv` | Données forestières auxiliaires utilisées pour construire les covariables associées aux conifères. |
| `data/Politic_divition/lpr_000b21a_e.shp` | Couche spatiale de limites utilisée pour le masquage et la visualisation du Québec. |

---

## Principales sorties

Pour chaque espèce, les sorties sont générées dans :

outputs/sugar/
outputs/red/

Sorties typiques :

| Sortie | Description |
|---|---|
| `training_table_*.csv` | Table d’entraînement au niveau de la grille. |
| `model_comparison_*.csv` | Comparaison des performances des modèles. |
| `cv_metrics_*.csv` | Métriques de validation croisée. |
| `present_*_probability.tif` | Rasters de pertinence écologique actuelle prédite. |
| `difference_*_env_minus_conifer.tif` | Rasters de différence entre le Modèle A et le Modèle B. |
| `feature_importance_*.csv` | Résumés de l’importance des variables. |
| `shap_summary_*` | Sorties associées à SHAP. |
| `future_projection_metrics_*.csv` | Métriques de pertinence écologique future par scénario. |
| `future_predictions/` | Rasters de probabilité de pertinence écologique future. |
| `stats/` | Sorties statistiques additionnelles, lorsqu’elles sont générées. |

---

## Prérequis

Python 3.10 ou version ultérieure est recommandé.

Principales dépendances Python :

geopandas
rasterio
numpy
pandas
scipy
scikit-learn
xgboost
shap
matplotlib

Installation suggérée :

pip install geopandas rasterio numpy pandas scipy scikit-learn xgboost shap matplotlib

Sur certains systèmes, notamment Windows, `geopandas` et `rasterio` peuvent nécessiter des dépendances GDAL et PROJ. Si l’installation avec `pip` échoue, il est préférable d’utiliser `conda` :

conda install -c conda-forge geopandas rasterio numpy pandas scipy scikit-learn xgboost shap matplotlib

---

## Démarrage rapide

### 1. Valider les entrées sans entraîner les modèles

python acer_grid_conifer_analysis.py --project-dir /chemin/vers/projet --species both --check-only

### 2. Exécuter l’analyse complète pour les deux espèces

python acer_grid_conifer_analysis.py --project-dir /chemin/vers/projet --species both

### 3. Exécuter l’analyse pour l’érable à sucre seulement

python acer_grid_conifer_analysis.py --project-dir /chemin/vers/projet --species sugar

### 4. Exécuter l’analyse pour l’érable rouge seulement

python acer_grid_conifer_analysis.py --project-dir /chemin/vers/projet --species red

---

## Arguments de ligne de commande

| Argument | Description |
|---|---|
| `--project-dir` | Répertoire de base du projet contenant les entrées et les dossiers de sortie. |
| `--species` | Espèce à analyser : `sugar`, `red` ou `both`. |
| `--check-only` | Exécute la validation des entrées sans entraîner les modèles ni générer de prédictions. |

---

## Cadre de modélisation

Le flux compare deux configurations de modèles :

### Modèle A — Environnemental

Ce modèle estime la pertinence écologique des érables en utilisant uniquement des prédicteurs environnementaux, incluant des variables climatiques et édaphiques.

### Modèle B — Environnemental + filtres associés aux conifères

Ce modèle étend le Modèle A en ajoutant des covariables associées à l’épinette noire, au pin gris et au sapin baumier. Ces covariables incluent :

- Des surfaces de densité par noyau.
- Des rasters de distance à la présence la plus proche.

L’objectif du Modèle B est d’évaluer si la matrice boréale de conifères améliore l’estimation de la pertinence écologique réalisée des érables, particulièrement près de leurs limites nordiques de distribution.

---

## Projections futures

La pertinence écologique future est projetée sous plusieurs combinaisons de modèles climatiques, de scénarios d’émission et de périodes temporelles. Le pipeline génère deux types de projections :

1. Projections futures environnementales  
   Le Modèle A est transféré vers les conditions environnementales futures.

2. Projections de sensibilité avec conifères statiques  
   Le Modèle B est transféré vers les conditions environnementales futures, en maintenant les covariables associées aux conifères fixes à leurs valeurs actuelles.

La configuration avec conifères statiques ne doit pas être interprétée comme une distribution forestière future réalisée. Il s’agit d’une analyse de sensibilité qui évalue comment la pertinence écologique future des érables change si la structure spatiale actuelle des conifères demeure fixe.

---

## Notes de reproductibilité

Le pipeline vise à rendre le flux de modélisation transparent et reproductible. Toutefois, les utilisateurs doivent vérifier :

- L’alignement des rasters.
- Les systèmes de coordonnées.
- La gestion des valeurs NoData.
- Les noms des variables d’entrée.
- Les conventions de nommage des scénarios futurs.
- La disponibilité et les licences des données de tiers.

Les fichiers raster volumineux et certaines sorties dérivées peuvent ne pas être inclus directement dans le dépôt GitHub. Lorsque nécessaire, ces fichiers doivent être archivés dans un dépôt de données comme Zenodo.

---

## Disponibilité des données

Le code d’analyse et le flux reproductible sont fournis dans ce dépôt. Les grands rasters d’entrée, les rasters dérivés et certaines sorties de modèles peuvent être archivés séparément dans Zenodo ou un autre dépôt public, selon les conditions de redistribution des données originales.

Certaines données d’entrée peuvent provenir de sources tierces et rester soumises à leurs conditions originales d’accès et de licence.

---

## Licence

Ce dépôt est distribué sous la licence indiquée dans le fichier `LICENSE`.
