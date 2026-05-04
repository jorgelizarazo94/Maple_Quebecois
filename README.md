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

## Estructura esperada de datos

El script espera una estructura de proyecto con carpetas como:

- `bioclim_data/recortados_alineados/` (rasters ambientales fuente).
- `data/db_sugar_maple.csv` y `data/db_red_maple.csv` (ocurrencias).
- `data/data_final_forestry_2.csv` (datos forestales auxiliares).
- `data/Politic_divition/lpr_000b21a_e.shp` (límite espacial).

Además, crea automáticamente carpetas de trabajo/salida, por ejemplo:

- `derived_rasters/`
- `model_inputs/`
- `models/`
- `outputs/sugar/`
- `outputs/red/`

## Requisitos

Python 3.10+ recomendado.

Dependencias principales:

- `geopandas`
- `rasterio`
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `xgboost`
- `shap`
- `matplotlib`

Instalación sugerida:

```bash
pip install geopandas rasterio numpy pandas scipy scikit-learn xgboost shap matplotlib
```

> Nota: en algunos sistemas `geopandas`/`rasterio` requieren bibliotecas del sistema (GDAL/PROJ).

## Uso rápido

### 1) Validar insumos sin entrenar

```bash
python acer_grid_conifer_analysis.py --project-dir /ruta/a/tu/proyecto --species both --check-only
```

### 2) Ejecutar análisis completo para ambas especies

```bash
python acer_grid_conifer_analysis.py --project-dir /ruta/a/tu/proyecto --species both
```

### 3) Ejecutar para una sola especie

```bash
python acer_grid_conifer_analysis.py --project-dir /ruta/a/tu/proyecto --species sugar
python acer_grid_conifer_analysis.py --project-dir /ruta/a/tu/proyecto --species red
```

## Parámetros CLI

- `--project-dir`: directorio base del proyecto con insumos y carpetas de salida.
- `--species`: `sugar`, `red` o `both`.
- `--check-only`: activa modo de validación de entradas.

## Salidas principales

Según la especie, el pipeline genera:

- CSVs de entrenamiento por grilla (`grid_training_*`).
- Modelos serializados (`xgb_*`).
- Comparaciones de desempeño (`model_comparison_*.csv`).
- Raster de probabilidad de presencia por tipo de modelo.
- Raster de diferencia entre modelos.
- Importancia de variables y gráfico SHAP.

## Licencia

Este repositorio se distribuye bajo la licencia incluida en `LICENSE`.
