# Maple_Quebecois

<p align="center">
  <img src="logos/Picture1.png" alt="Institute of Chartered Foresters logo" width="520"/>
</p>

---

## English

### Project overview
Geospatial pipeline to model the potential distribution of **Acer saccharum** (sugar maple) and **Acer rubrum** (red maple) in Québec, combining bioclimatic predictors and conifer-related covariates.

### Repository contents
- `acer_grid_conifer_analysis.py`: main reproducible workflow (input checks, training, prediction, outputs).
- `Supplementary_*.ipynb`: supplementary analyses and result notebooks.
- `rasters_futuro.ipynb`: future/scenario raster exploration.
- `logos/`: graphic assets.

### What the pipeline does
1. Repairs/projects environmental rasters to an analysis CRS.
2. Builds conifer covariates (kernel density + distance rasters).
3. Creates species response/presence layers.
4. Trains and evaluates two model families per species:
   - Environmental-only model.
   - Environmental + conifer model.
5. Exports probability maps, model-difference rasters, and feature-importance tables.

### Expected data layout
- `bioclim_data/recortados_alineados/` (source environmental rasters).
- `data/db_sugar_maple.csv` and `data/db_red_maple.csv` (occurrences).
- `data/data_final_forestry_2.csv` (forestry covariates/support data).
- `data/Politic_divition/lpr_000b21a_e.shp` (boundary shapefile).

The script also creates:
- `derived_rasters/`
- `model_inputs/`
- `models/`
- `outputs/sugar/`
- `outputs/red/`

### Requirements
Python 3.10+ recommended.

Main dependencies:
`geopandas rasterio numpy pandas scipy scikit-learn xgboost shap matplotlib`

Install:
```bash
pip install geopandas rasterio numpy pandas scipy scikit-learn xgboost shap matplotlib
```

### Quick start
Validate only:
```bash
python acer_grid_conifer_analysis.py --project-dir /path/to/project --species both --check-only
```

Run full pipeline (both species):
```bash
python acer_grid_conifer_analysis.py --project-dir /path/to/project --species both
```

Run a single species:
```bash
python acer_grid_conifer_analysis.py --project-dir /path/to/project --species sugar
python acer_grid_conifer_analysis.py --project-dir /path/to/project --species red
```

CLI parameters:
- `--project-dir`: project root with inputs/outputs.
- `--species`: `sugar`, `red`, or `both`.
- `--check-only`: run validations without fitting/exporting final model outputs.

Main outputs:
- Grid training CSVs (`grid_training_*`).
- Serialized models (`xgb_*`).
- Performance comparison tables (`model_comparison_*.csv`).
- Presence probability rasters by model.
- Difference rasters between models.
- Feature-importance CSVs and SHAP summary figures.

---

## Français (Québécois)

### Aperçu du projet
Pipeline géospatial pour modéliser la distribution potentielle de **l’érable à sucre** (*Acer saccharum*) et de **l’érable rouge** (*Acer rubrum*) au Québec, en combinant des prédicteurs bioclimatiques et des covariables liées aux conifères.

### Contenu du dépôt
- `acer_grid_conifer_analysis.py` : flux principal reproductible (validation, entraînement, prédiction, sorties).
- `Supplementary_*.ipynb` : cahiers d’analyses complémentaires.
- `rasters_futuro.ipynb` : exploration de rasters/scénarios futurs.
- `logos/` : ressources graphiques.

### Ce que fait le pipeline
1. Corrige/projette les rasters environnementaux vers un CRS d’analyse.
2. Génère des covariables de conifères (noyaux de densité + distances).
3. Produit des couches de réponse/présence.
4. Entraîne et compare deux familles de modèles par espèce :
   - Modèle environnemental seulement.
   - Modèle environnemental + conifères.
5. Exporte des cartes de probabilité, des rasters de différence et des tableaux d’importance des variables.

### Structure de données attendue
- `bioclim_data/recortados_alineados/`
- `data/db_sugar_maple.csv` et `data/db_red_maple.csv`
- `data/data_final_forestry_2.csv`
- `data/Politic_divition/lpr_000b21a_e.shp`

Dossiers générés automatiquement :
- `derived_rasters/`, `model_inputs/`, `models/`, `outputs/sugar/`, `outputs/red/`

### Exigences
Python 3.10+ recommandé.

Dépendances principales :
`geopandas rasterio numpy pandas scipy scikit-learn xgboost shap matplotlib`

Installation :
```bash
pip install geopandas rasterio numpy pandas scipy scikit-learn xgboost shap matplotlib
```

### Démarrage rapide
Validation seulement :
```bash
python acer_grid_conifer_analysis.py --project-dir /chemin/du/projet --species both --check-only
```

Exécution complète :
```bash
python acer_grid_conifer_analysis.py --project-dir /chemin/du/projet --species both
```

Paramètres CLI :
- `--project-dir`, `--species`, `--check-only`

Sorties principales :
- CSV d’entraînement, modèles sérialisés, tableaux de performance,
- rasters de probabilité et de différence,
- importances de variables et figures SHAP.

---

## Español

### Descripción del proyecto
Pipeline geoespacial para modelar la distribución potencial de **Acer saccharum** (arce de azúcar) y **Acer rubrum** (arce rojo) en Québec, integrando predictores bioclimáticos y covariables asociadas a coníferas.

### Contenido del repositorio
- `acer_grid_conifer_analysis.py`: flujo principal reproducible (validación, entrenamiento, predicción y salidas).
- `Supplementary_*.ipynb`: análisis complementarios.
- `rasters_futuro.ipynb`: exploración de escenarios raster.
- `logos/`: recursos gráficos.

### Qué hace el pipeline
1. Repara/proyecta rasters ambientales al CRS de análisis.
2. Construye covariables de coníferas (kernels + distancias).
3. Genera capas de respuesta/presencia.
4. Entrena dos familias de modelos por especie:
   - Solo ambiente.
   - Ambiente + coníferas.
5. Exporta mapas de probabilidad, diferencias e importancia de variables.

### Estructura esperada
- `bioclim_data/recortados_alineados/`
- `data/db_sugar_maple.csv` y `data/db_red_maple.csv`
- `data/data_final_forestry_2.csv`
- `data/Politic_divition/lpr_000b21a_e.shp`

Salidas generadas:
- `derived_rasters/`, `model_inputs/`, `models/`, `outputs/sugar/`, `outputs/red/`

### Requisitos e instalación
Python 3.10+.

```bash
pip install geopandas rasterio numpy pandas scipy scikit-learn xgboost shap matplotlib
```

### Uso rápido
```bash
python acer_grid_conifer_analysis.py --project-dir /ruta/al/proyecto --species both --check-only
python acer_grid_conifer_analysis.py --project-dir /ruta/al/proyecto --species both
python acer_grid_conifer_analysis.py --project-dir /ruta/al/proyecto --species sugar
python acer_grid_conifer_analysis.py --project-dir /ruta/al/proyecto --species red
```

### Licencia
Este repositorio se distribuye bajo `LICENSE`.
