from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio
import shap
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.transform import rowcol, xy
from rasterio.warp import Resampling, calculate_default_transform, reproject
from scipy.ndimage import distance_transform_edt, gaussian_filter
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ENVIRONMENTAL_PREDICTORS = [
    "bio15",
    "ph",
    "bio7",
    "bio3",
    "bio31",
    "sand",
    "bio34",
    "bio12",
    "bio23",
    "bio1",
]

CONIFER_SPECIES = {
    "EN": "Picea_mariana",
    "PG": "Pinus_banksiana",
    "SB": "Abies_balsamea",
}

CONIFER_KERNEL_PREDICTORS = [
    "Picea_mariana_kernel_sigma5",
    "Pinus_banksiana_kernel_sigma5",
    "Abies_balsamea_kernel_sigma5",
]

CONIFER_DISTANCE_PREDICTORS = [
    "Picea_mariana_distance_km",
    "Pinus_banksiana_distance_km",
    "Abies_balsamea_distance_km",
]

MODEL_B_PREDICTORS = ENVIRONMENTAL_PREDICTORS + CONIFER_KERNEL_PREDICTORS + CONIFER_DISTANCE_PREDICTORS

MODEL_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 400,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "random_state": 42,
}

BACKGROUND_RATIO = 5
RANDOM_SEED = 42
BLOCK_SIZE = 50
PREDICTION_NODATA = -9999.0
RESPONSE_NODATA = 255
SOURCE_ENVIRONMENT_CRS = CRS.from_epsg(4326)
TARGET_ANALYSIS_CRS = CRS.from_epsg(32198)


class DataValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProjectPaths:
    project_dir: Path
    source_env_dir: Path
    env_dir: Path
    sugar_occurrence_csv: Path
    red_occurrence_csv: Path
    forestry_csv: Path
    boundary_shapefile: Path
    source_reference_raster: Path
    reference_raster: Path
    derived_rasters_dir: Path
    repaired_env_dir: Path
    conifer_covariates_dir: Path
    sugar_response_dir: Path
    red_response_dir: Path
    model_inputs_dir: Path
    models_dir: Path
    outputs_dir: Path
    sugar_outputs_dir: Path
    red_outputs_dir: Path

    @classmethod
    def from_project_dir(cls, project_dir: str | Path) -> "ProjectPaths":
        project_path = Path(project_dir)
        derived_dir = project_path / "derived_rasters"
        outputs_dir = project_path / "outputs"
        repaired_env_dir = derived_dir / "environment_projected_32198"
        return cls(
            project_dir=project_path,
            source_env_dir=project_path / "bioclim_data" / "recortados_alineados",
            env_dir=repaired_env_dir,
            sugar_occurrence_csv=project_path / "data" / "db_sugar_maple.csv",
            red_occurrence_csv=project_path / "data" / "db_red_maple.csv",
            forestry_csv=project_path / "data" / "data_final_forestry_2.csv",
            boundary_shapefile=project_path / "data" / "Politic_divition" / "lpr_000b21a_e.shp",
            source_reference_raster=project_path / "bioclim_data" / "recortados_alineados" / "bio1.tif",
            reference_raster=repaired_env_dir / "bio1.tif",
            derived_rasters_dir=derived_dir,
            repaired_env_dir=repaired_env_dir,
            conifer_covariates_dir=derived_dir / "conifer_covariates",
            sugar_response_dir=derived_dir / "sugar_response",
            red_response_dir=derived_dir / "red_response",
            model_inputs_dir=project_path / "model_inputs",
            models_dir=project_path / "models",
            outputs_dir=outputs_dir,
            sugar_outputs_dir=outputs_dir / "sugar",
            red_outputs_dir=outputs_dir / "red",
        )


@dataclass(frozen=True)
class ReferenceGrid:
    path: Path
    crs: Any
    transform: Any
    width: int
    height: int
    resolution: tuple[float, float]
    nodata: float | int | None
    valid_mask: np.ndarray
    profile: dict[str, Any]


SPECIES_CONFIG = {
    "sugar": {
        "species": "sugar",
        "display_name": "Sugar maple",
        "occurrence_csv": "sugar_occurrence_csv",
        "response_column": "sugar_presence",
        "response_raster": "sugar_maple_presence_grid.tif",
        "env_training_csv": "grid_training_sugar_env_only.csv",
        "conifer_training_csv": "grid_training_sugar_env_conifer.csv",
        "env_model": "xgb_sugar_env_only.pkl",
        "conifer_model": "xgb_sugar_env_conifer.pkl",
        "comparison_csv": "model_comparison_sugar.csv",
        "env_probability": "present_sugar_env_only_probability.tif",
        "conifer_probability": "present_sugar_env_conifer_probability.tif",
        "difference_raster": "difference_sugar_env_minus_conifer.tif",
        "env_importance": "feature_importance_sugar_env_only.csv",
        "conifer_importance": "feature_importance_sugar_env_conifer.csv",
        "conifer_shap": "shap_summary_sugar_env_conifer.png",
        "outputs_dir": "sugar_outputs_dir",
        "response_dir": "sugar_response_dir",
    },
    "red": {
        "species": "red",
        "display_name": "Red maple",
        "occurrence_csv": "red_occurrence_csv",
        "response_column": "red_presence",
        "response_raster": "red_maple_presence_grid.tif",
        "env_training_csv": "grid_training_red_env_only.csv",
        "conifer_training_csv": "grid_training_red_env_conifer.csv",
        "env_model": "xgb_red_env_only.pkl",
        "conifer_model": "xgb_red_env_conifer.pkl",
        "comparison_csv": "model_comparison_red.csv",
        "env_probability": "present_red_env_only_probability.tif",
        "conifer_probability": "present_red_env_conifer_probability.tif",
        "difference_raster": "difference_red_env_minus_conifer.tif",
        "env_importance": "feature_importance_red_env_only.csv",
        "conifer_importance": "feature_importance_red_env_conifer.csv",
        "conifer_shap": "shap_summary_red_env_conifer.png",
        "outputs_dir": "red_outputs_dir",
        "response_dir": "red_response_dir",
    },
}


def ensure_output_directories(paths: ProjectPaths) -> None:
    required_dirs = [
        paths.derived_rasters_dir,
        paths.repaired_env_dir,
        paths.conifer_covariates_dir,
        paths.sugar_response_dir,
        paths.red_response_dir,
        paths.model_inputs_dir,
        paths.models_dir,
        paths.outputs_dir,
        paths.sugar_outputs_dir,
        paths.red_outputs_dir,
    ]
    for required_dir in required_dirs:
        required_dir.mkdir(parents=True, exist_ok=True)


def load_reference_grid(reference_path: Path) -> ReferenceGrid:
    with rasterio.open(reference_path) as src:
        reference_array = src.read(1, masked=True)
        valid_mask = ~np.ma.getmaskarray(reference_array)
        if not valid_mask.any():
            if src.nodata is None or (isinstance(src.nodata, float) and np.isnan(src.nodata)):
                valid_mask = np.isfinite(reference_array.filled(np.nan))
            else:
                valid_mask = reference_array.filled(src.nodata) != src.nodata

        profile = src.profile.copy()

        return ReferenceGrid(
            path=reference_path,
            crs=src.crs,
            transform=src.transform,
            width=src.width,
            height=src.height,
            resolution=src.res,
            nodata=src.nodata,
            valid_mask=valid_mask.astype(bool),
            profile=profile,
        )


def source_environment_raster_paths(paths: ProjectPaths) -> dict[str, Path]:
    return {name: paths.source_env_dir / f"{name}.tif" for name in ENVIRONMENTAL_PREDICTORS}


def environmental_raster_paths(paths: ProjectPaths) -> dict[str, Path]:
    return {name: paths.env_dir / f"{name}.tif" for name in ENVIRONMENTAL_PREDICTORS}


def ensure_projected_environment_stack(paths: ProjectPaths) -> dict[str, Any]:
    source_paths = source_environment_raster_paths(paths)
    missing = [name for name, path in source_paths.items() if not path.exists()]
    if missing:
        raise DataValidationError(f"Missing source environmental rasters: {', '.join(missing)}")

    repaired_paths = environmental_raster_paths(paths)
    if all(path.exists() for path in repaired_paths.values()) and paths.reference_raster.exists():
        try:
            reference = load_reference_grid(paths.reference_raster)
            if not reference.valid_mask.any():
                raise DataValidationError("Existing repaired reference raster has no valid cells.")
            validate_environment_rasters(paths, reference, strict=True)
            ensure_projected_distance_support(reference)
            return {
                "status": "loaded_existing",
                "created": False,
                "target_crs": str(reference.crs),
                "reference_raster": str(paths.reference_raster),
            }
        except DataValidationError:
            pass

    with rasterio.open(paths.source_reference_raster) as source_reference:
        source_transform = source_reference.transform
        source_width = source_reference.width
        source_height = source_reference.height
        source_bounds = source_reference.bounds
        source_crs = source_reference.crs or SOURCE_ENVIRONMENT_CRS

        if source_reference.crs is not None and source_reference.crs != SOURCE_ENVIRONMENT_CRS:
            raise DataValidationError(
                f"Unsupported source reference CRS {source_reference.crs}; expected geographic {SOURCE_ENVIRONMENT_CRS}."
            )

        target_transform, target_width, target_height = calculate_default_transform(
            source_crs,
            TARGET_ANALYSIS_CRS,
            source_width,
            source_height,
            *source_bounds,
        )

    target_profile = {
        "driver": "GTiff",
        "height": target_height,
        "width": target_width,
        "count": 1,
        "dtype": "float32",
        "crs": TARGET_ANALYSIS_CRS,
        "transform": target_transform,
        "nodata": np.nan,
        "compress": "lzw",
    }

    paths.repaired_env_dir.mkdir(parents=True, exist_ok=True)

    for raster_name, source_path in source_paths.items():
        with rasterio.open(source_path) as src:
            effective_crs = src.crs or SOURCE_ENVIRONMENT_CRS
            if effective_crs != SOURCE_ENVIRONMENT_CRS:
                raise DataValidationError(
                    f"Unsupported source CRS for {source_path.name}: {effective_crs}; expected {SOURCE_ENVIRONMENT_CRS}."
                )

            aligned = (
                src.transform == source_transform
                and src.width == source_width
                and src.height == source_height
            )
            if not aligned:
                raise DataValidationError(
                    f"Source raster alignment mismatch for {source_path.name}: "
                    f"transform={src.transform}, shape=({src.width}, {src.height}) versus "
                    f"reference transform={source_transform}, shape=({source_width}, {source_height})"
                )

            source_array = src.read(1).astype(np.float32)
            destination = np.full((target_height, target_width), np.nan, dtype=np.float32)
            reproject(
                source=source_array,
                destination=destination,
                src_transform=src.transform,
                src_crs=effective_crs,
                src_nodata=src.nodata,
                dst_transform=target_transform,
                dst_crs=TARGET_ANALYSIS_CRS,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )

            output_path = repaired_paths[raster_name]
            with rasterio.open(output_path, "w", **target_profile) as dst:
                dst.write(destination, 1)

    repaired_reference = load_reference_grid(paths.reference_raster)
    validate_environment_rasters(paths, repaired_reference, strict=True)
    ensure_projected_distance_support(repaired_reference)

    return {
        "status": "created",
        "created": True,
        "target_crs": str(TARGET_ANALYSIS_CRS),
        "reference_raster": str(paths.reference_raster),
        "reference_width": repaired_reference.width,
        "reference_height": repaired_reference.height,
        "reference_resolution": repaired_reference.resolution,
    }


def conifer_covariate_paths(paths: ProjectPaths) -> dict[str, Path]:
    output_paths: dict[str, Path] = {}
    for species_name in CONIFER_SPECIES.values():
        output_paths[f"{species_name}_presence"] = paths.conifer_covariates_dir / f"{species_name}_presence.tif"
        output_paths[f"{species_name}_count"] = paths.conifer_covariates_dir / f"{species_name}_count.tif"
        output_paths[f"{species_name}_kernel_sigma5"] = paths.conifer_covariates_dir / f"{species_name}_kernel_sigma5.tif"
        output_paths[f"{species_name}_distance_km"] = paths.conifer_covariates_dir / f"{species_name}_distance_km.tif"
    return output_paths


def format_raster_mismatch(path: Path, reference: ReferenceGrid) -> str:
    with rasterio.open(path) as src:
        return (
            f"{path.name}: crs={src.crs}, transform={src.transform}, "
            f"shape=({src.width}, {src.height}) versus "
            f"reference crs={reference.crs}, transform={reference.transform}, "
            f"shape=({reference.width}, {reference.height})"
        )


def validate_environment_rasters(
    paths: ProjectPaths,
    reference: ReferenceGrid,
    strict: bool = True,
) -> dict[str, Any]:
    available: list[str] = []
    missing: list[str] = []
    mismatches: list[str] = []

    for raster_name, raster_path in environmental_raster_paths(paths).items():
        if not raster_path.exists():
            missing.append(raster_name)
            continue

        available.append(raster_name)
        with rasterio.open(raster_path) as src:
            aligned = (
                src.crs == reference.crs
                and src.transform == reference.transform
                and src.width == reference.width
                and src.height == reference.height
            )
        if not aligned:
            mismatches.append(format_raster_mismatch(raster_path, reference))

    report = {
        "reference_crs": str(reference.crs),
        "reference_transform": tuple(reference.transform),
        "reference_width": reference.width,
        "reference_height": reference.height,
        "reference_resolution": reference.resolution,
        "reference_nodata": reference.nodata,
        "available_rasters": available,
        "missing_rasters": missing,
        "alignment_status": "aligned" if not missing and not mismatches else "failed",
        "mismatched_rasters": mismatches,
    }

    if strict and (missing or mismatches):
        messages = []
        if missing:
            messages.append(f"Missing environmental rasters: {', '.join(missing)}")
        if mismatches:
            messages.append("Alignment/CRS mismatches: " + " | ".join(mismatches))
        raise DataValidationError("; ".join(messages))

    return report


def load_occurrence_points(csv_path: Path, reference: ReferenceGrid) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    df = pd.read_csv(csv_path)
    required_columns = {"longitude", "latitude"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise DataValidationError(f"Missing columns in {csv_path.name}: {sorted(missing_columns)}")

    original_records = len(df)
    valid_coordinate_mask = df["longitude"].notna() & df["latitude"].notna()
    valid_df = df.loc[valid_coordinate_mask].copy()

    if reference.crs is None:
        raise DataValidationError(
            f"Reference raster {reference.path.name} has no CRS. Points cannot be reprojected to the reference grid safely."
        )

    geodf = gpd.GeoDataFrame(
        valid_df,
        geometry=gpd.points_from_xy(valid_df["longitude"], valid_df["latitude"]),
        crs="EPSG:4326",
    ).to_crs(reference.crs)

    report = {
        "original_records": int(original_records),
        "valid_coordinate_records": int(valid_coordinate_mask.sum()),
        "removed_records": int(original_records - valid_coordinate_mask.sum()),
    }
    return geodf, report


def write_array_as_raster(
    array: np.ndarray,
    reference: ReferenceGrid,
    output_path: Path,
    dtype: str,
    nodata: float | int | None,
) -> None:
    output_profile = reference.profile.copy()
    output_profile.update(
        count=1,
        dtype=dtype,
        nodata=nodata,
        compress="lzw",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **output_profile) as dst:
        dst.write(array.astype(dtype), 1)


def rasterize_response(
    points: gpd.GeoDataFrame,
    reference: ReferenceGrid,
    output_path: Path,
) -> dict[str, int | str]:
    rasterized = rasterize(
        ((geom, 1) for geom in points.geometry if geom is not None and not geom.is_empty),
        out_shape=(reference.height, reference.width),
        transform=reference.transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    )

    x_values = points.geometry.x.to_numpy()
    y_values = points.geometry.y.to_numpy()
    rows, cols = rowcol(reference.transform, x_values, y_values)
    in_bounds_mask = (
        (rows >= 0)
        & (rows < reference.height)
        & (cols >= 0)
        & (cols < reference.width)
    )

    valid_rows = rows[in_bounds_mask]
    valid_cols = cols[in_bounds_mask]
    occupied_cells = set(zip(valid_rows.tolist(), valid_cols.tolist()))

    rasterized_with_nodata = np.where(reference.valid_mask, rasterized, RESPONSE_NODATA).astype(np.uint8)
    write_array_as_raster(rasterized_with_nodata, reference, output_path, dtype="uint8", nodata=RESPONSE_NODATA)

    return {
        "output_raster": str(output_path),
        "original_points": int(len(points)),
        "occupied_raster_cells": int(len(occupied_cells)),
        "duplicate_points_collapsed": int(in_bounds_mask.sum() - len(occupied_cells)),
        "points_outside_grid": int((~in_bounds_mask).sum()),
    }


def read_raster_array(raster_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with rasterio.open(raster_path) as src:
        masked = src.read(1, masked=True)
        valid_mask = ~np.ma.getmaskarray(masked)
        fill_value = src.nodata if src.nodata is not None else 0
        array = masked.filled(fill_value).astype(np.float32)
        array[~valid_mask] = np.nan
        info = {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "nodata": src.nodata,
        }
    return array, valid_mask.astype(bool), info


def validate_existing_raster_alignment(raster_path: Path, reference: ReferenceGrid) -> None:
    _, _, info = read_raster_array(raster_path)
    aligned = (
        info["crs"] == reference.crs
        and info["transform"] == reference.transform
        and info["width"] == reference.width
        and info["height"] == reference.height
    )
    if not aligned:
        raise DataValidationError(format_raster_mismatch(raster_path, reference))


def build_count_raster(points: gpd.GeoDataFrame, reference: ReferenceGrid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_values = points.geometry.x.to_numpy()
    y_values = points.geometry.y.to_numpy()
    rows, cols = rowcol(reference.transform, x_values, y_values)

    in_bounds_mask = (
        (rows >= 0)
        & (rows < reference.height)
        & (cols >= 0)
        & (cols < reference.width)
    )
    rows = rows[in_bounds_mask]
    cols = cols[in_bounds_mask]

    count_raster = np.zeros((reference.height, reference.width), dtype=np.uint32)
    np.add.at(count_raster, (rows, cols), 1)

    return count_raster, rows, cols


def ensure_projected_distance_support(reference: ReferenceGrid) -> None:
    if reference.crs is None:
        raise DataValidationError(
            f"Reference raster {reference.path.name} has no CRS. Distance rasters cannot be calculated safely."
        )

    if reference.crs.is_geographic:
        raise DataValidationError(
            f"Reference raster {reference.path.name} is in geographic degrees ({reference.crs}). "
            "Distance rasters cannot be calculated safely in degrees."
        )

    if not reference.crs.is_projected:
        raise DataValidationError(
            f"Reference raster {reference.path.name} is not projected ({reference.crs}). "
            "Distance rasters require projected linear units."
        )


def ensure_conifer_covariates(
    paths: ProjectPaths,
    reference: ReferenceGrid,
    allow_create: bool = True,
) -> tuple[dict[str, Path], dict[str, Any]]:
    covariate_paths = conifer_covariate_paths(paths)
    all_outputs_exist = all(path.exists() for path in covariate_paths.values())
    if all_outputs_exist:
        try:
            for raster_path in covariate_paths.values():
                validate_existing_raster_alignment(raster_path, reference)
            return covariate_paths, {"status": "loaded_existing", "created": False}
        except DataValidationError:
            for raster_path in covariate_paths.values():
                if raster_path.exists():
                    raster_path.unlink()

    if not allow_create:
        raise DataValidationError("Conifer covariates are missing and creation is disabled for this run.")

    ensure_projected_distance_support(reference)

    forestry_df = pd.read_csv(paths.forestry_csv)
    required_columns = {"longitude", "latitude", "essence"}
    missing_columns = required_columns.difference(forestry_df.columns)
    if missing_columns:
        raise DataValidationError(f"Missing columns in {paths.forestry_csv.name}: {sorted(missing_columns)}")

    valid_forestry_mask = forestry_df["longitude"].notna() & forestry_df["latitude"].notna()
    forestry_df = forestry_df.loc[valid_forestry_mask].copy()

    forestry_gdf = gpd.GeoDataFrame(
        forestry_df,
        geometry=gpd.points_from_xy(forestry_df["longitude"], forestry_df["latitude"]),
        crs="EPSG:4326",
    ).to_crs(reference.crs)

    pixel_size_y = abs(reference.transform.e)
    pixel_size_x = abs(reference.transform.a)

    report: dict[str, Any] = {
        "status": "created",
        "created": True,
        "species": {},
    }

    for species_code, species_name in CONIFER_SPECIES.items():
        species_points = forestry_gdf.loc[forestry_gdf["essence"] == species_code].copy()
        count_raster, rows, cols = build_count_raster(species_points, reference)
        presence_raster = (count_raster > 0).astype(np.uint8)

        kernel_raster = gaussian_filter(count_raster.astype(np.float32), sigma=5.0, mode="constant")
        kernel_valid = kernel_raster[reference.valid_mask]
        if kernel_valid.size and float(kernel_valid.max()) > float(kernel_valid.min()):
            kernel_raster = (kernel_raster - kernel_valid.min()) / (kernel_valid.max() - kernel_valid.min())
        else:
            kernel_raster = np.zeros_like(kernel_raster, dtype=np.float32)

        distance_raster = distance_transform_edt(
            presence_raster == 0,
            sampling=(pixel_size_y, pixel_size_x),
        ).astype(np.float32) / 1000.0

        presence_output = covariate_paths[f"{species_name}_presence"]
        count_output = covariate_paths[f"{species_name}_count"]
        kernel_output = covariate_paths[f"{species_name}_kernel_sigma5"]
        distance_output = covariate_paths[f"{species_name}_distance_km"]

        write_array_as_raster(
            np.where(reference.valid_mask, presence_raster, RESPONSE_NODATA),
            reference,
            presence_output,
            dtype="uint8",
            nodata=RESPONSE_NODATA,
        )
        write_array_as_raster(
            np.where(reference.valid_mask, count_raster, 0),
            reference,
            count_output,
            dtype="uint32",
            nodata=0,
        )
        write_array_as_raster(
            np.where(reference.valid_mask, kernel_raster, PREDICTION_NODATA),
            reference,
            kernel_output,
            dtype="float32",
            nodata=PREDICTION_NODATA,
        )
        write_array_as_raster(
            np.where(reference.valid_mask, distance_raster, PREDICTION_NODATA),
            reference,
            distance_output,
            dtype="float32",
            nodata=PREDICTION_NODATA,
        )

        report["species"][species_name] = {
            "records": int(len(species_points)),
            "occupied_cells": int(np.count_nonzero(presence_raster)),
            "duplicates_collapsed": int(len(rows) - np.count_nonzero(presence_raster)),
        }

    return covariate_paths, report


def assemble_predictor_arrays(
    raster_paths: dict[str, Path],
    reference: ReferenceGrid,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    combined_valid_mask = reference.valid_mask.copy()

    for predictor_name, raster_path in raster_paths.items():
        validate_existing_raster_alignment(raster_path, reference)
        array, valid_mask, _ = read_raster_array(raster_path)
        arrays[predictor_name] = array
        combined_valid_mask &= valid_mask & np.isfinite(array)

    return arrays, combined_valid_mask


def build_training_tables(
    species_key: str,
    paths: ProjectPaths,
    reference: ReferenceGrid,
    response_raster_path: Path,
    conifer_paths: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    config = SPECIES_CONFIG[species_key]
    response_column = config["response_column"]

    env_arrays, env_valid_mask = assemble_predictor_arrays(environmental_raster_paths(paths), reference)
    conifer_predictor_paths = {
        predictor: conifer_paths[predictor] for predictor in CONIFER_KERNEL_PREDICTORS + CONIFER_DISTANCE_PREDICTORS
    }
    conifer_arrays, conifer_valid_mask = assemble_predictor_arrays(conifer_predictor_paths, reference)

    response_array, response_valid_mask, _ = read_raster_array(response_raster_path)
    response_binary = np.where(response_array == 1, 1, 0).astype(np.uint8)

    model_b_valid_mask = env_valid_mask & conifer_valid_mask & response_valid_mask
    model_b_valid_mask &= np.isfinite(response_array)

    presence_locations = np.argwhere((response_binary == 1) & model_b_valid_mask)
    background_locations = np.argwhere((response_binary == 0) & model_b_valid_mask)

    if len(presence_locations) == 0:
        raise DataValidationError(f"No presence cells were found for {config['display_name']} after rasterisation.")

    requested_background = len(presence_locations) * BACKGROUND_RATIO
    if len(background_locations) < requested_background:
        raise DataValidationError(
            f"Requested {requested_background} background cells but only {len(background_locations)} valid background cells are available."
        )

    rng = np.random.default_rng(RANDOM_SEED)
    background_choice = rng.choice(len(background_locations), size=requested_background, replace=False)
    sampled_background = background_locations[background_choice]

    sampled_locations = np.vstack([presence_locations, sampled_background])
    sampled_rows = sampled_locations[:, 0]
    sampled_cols = sampled_locations[:, 1]

    sampled_x, sampled_y = xy(reference.transform, sampled_rows, sampled_cols)

    combined_df = pd.DataFrame(
        {
            "row": sampled_rows,
            "col": sampled_cols,
            "x": sampled_x,
            "y": sampled_y,
            response_column: response_binary[sampled_rows, sampled_cols],
        }
    )

    for predictor_name, array in env_arrays.items():
        combined_df[predictor_name] = array[sampled_rows, sampled_cols]

    for predictor_name, array in conifer_arrays.items():
        combined_df[predictor_name] = array[sampled_rows, sampled_cols]

    combined_df = add_spatial_blocks(combined_df)
    combined_df.insert(0, "sample_id", np.arange(len(combined_df)))

    env_df = combined_df[["sample_id", "row", "col", "x", "y", response_column, *ENVIRONMENTAL_PREDICTORS, "spatial_block"]].copy()
    conifer_df = combined_df[
        ["sample_id", "row", "col", "x", "y", response_column, *MODEL_B_PREDICTORS, "spatial_block"]
    ].copy()

    env_output = paths.model_inputs_dir / config["env_training_csv"]
    conifer_output = paths.model_inputs_dir / config["conifer_training_csv"]
    env_df.to_csv(env_output, index=False)
    conifer_df.to_csv(conifer_output, index=False)

    report = describe_training_table(conifer_df, response_column)
    report.update(
        {
            "env_training_csv": str(env_output),
            "conifer_training_csv": str(conifer_output),
            "presence_cells": int((combined_df[response_column] == 1).sum()),
            "background_cells": int((combined_df[response_column] == 0).sum()),
            "same_sample_ids_for_models": bool(env_df["sample_id"].equals(conifer_df["sample_id"])),
        }
    )

    return env_df, conifer_df, report


def add_spatial_blocks(df: pd.DataFrame, block_size: int = BLOCK_SIZE) -> pd.DataFrame:
    result = df.copy()
    result["block_row"] = (result["row"] // block_size).astype(int)
    result["block_col"] = (result["col"] // block_size).astype(int)
    result["spatial_block"] = result["block_row"].astype(str) + "_" + result["block_col"].astype(str)
    result = result.drop(columns=["block_row", "block_col"])
    return result


def describe_training_table(df: pd.DataFrame, response_column: str) -> dict[str, Any]:
    block_counts = df["spatial_block"].value_counts()
    presence_count = int((df[response_column] == 1).sum())
    background_count = int((df[response_column] == 0).sum())
    return {
        "unique_blocks": int(block_counts.size),
        "minimum_samples_per_block": int(block_counts.min()),
        "maximum_samples_per_block": int(block_counts.max()),
        "presence_background_ratio": f"{presence_count}:{background_count}",
    }


def metric_or_nan(metric_func: Any, y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(metric_func(y_true, y_score))


def evaluate_model(
    df: pd.DataFrame,
    feature_names: list[str],
    response_column: str,
    model_output_path: Path,
) -> tuple[XGBClassifier, pd.DataFrame, dict[str, Any]]:
    groups = df["spatial_block"]
    unique_groups = groups.nunique()
    if unique_groups < 2:
        raise DataValidationError("At least two spatial blocks are required for GroupKFold cross-validation.")

    n_splits = min(5, unique_groups)
    group_kfold = GroupKFold(n_splits=n_splits)
    X = df[feature_names]
    y = df[response_column].astype(int)

    fold_rows: list[dict[str, Any]] = []
    for fold_number, (train_index, test_index) in enumerate(group_kfold.split(X, y, groups), start=1):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        model = XGBClassifier(**MODEL_PARAMS)
        model.fit(X_train, y_train)

        y_probability = model.predict_proba(X_test)[:, 1]
        y_prediction = (y_probability >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, y_prediction, labels=[0, 1]).ravel()
        sensitivity = float(tp / (tp + fn)) if (tp + fn) else float("nan")
        specificity = float(tn / (tn + fp)) if (tn + fp) else float("nan")
        tss = float(sensitivity + specificity - 1.0) if np.isfinite(sensitivity) and np.isfinite(specificity) else float("nan")

        fold_rows.append(
            {
                "fold": fold_number,
                "roc_auc": metric_or_nan(roc_auc_score, y_test.to_numpy(), y_probability),
                "pr_auc": metric_or_nan(average_precision_score, y_test.to_numpy(), y_probability),
                "sensitivity": sensitivity,
                "specificity": specificity,
                "tss": tss,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )

    fold_metrics = pd.DataFrame(fold_rows)
    summary = {
        "mean_ROC_AUC": float(fold_metrics["roc_auc"].mean()),
        "sd_ROC_AUC": float(fold_metrics["roc_auc"].std(ddof=1)) if len(fold_metrics) > 1 else 0.0,
        "mean_PR_AUC": float(fold_metrics["pr_auc"].mean()),
        "sd_PR_AUC": float(fold_metrics["pr_auc"].std(ddof=1)) if len(fold_metrics) > 1 else 0.0,
        "mean_sensitivity": float(fold_metrics["sensitivity"].mean()),
        "mean_specificity": float(fold_metrics["specificity"].mean()),
        "mean_TSS": float(fold_metrics["tss"].mean()),
    }

    final_model = XGBClassifier(**MODEL_PARAMS)
    final_model.fit(X, y)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_output_path, "wb") as handle:
        pickle.dump({"model": final_model, "feature_names": feature_names}, handle)

    return final_model, fold_metrics, summary


def compute_feature_importance(
    model: XGBClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    output_csv: Path,
    shap_output_png: Path | None = None,
) -> pd.DataFrame:
    booster = model.get_booster()
    gain_importance = booster.get_score(importance_type="gain")
    weight_importance = booster.get_score(importance_type="weight")

    permutation_sample = X.sample(min(len(X), 5000), random_state=RANDOM_SEED)
    permutation_target = y.loc[permutation_sample.index]
    if len(np.unique(permutation_target)) > 1:
        permutation = permutation_importance(
            model,
            permutation_sample,
            permutation_target,
            scoring="roc_auc",
            n_repeats=10,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        permutation_mean = permutation.importances_mean
        permutation_std = permutation.importances_std
    else:
        permutation_mean = np.full(X.shape[1], np.nan)
        permutation_std = np.full(X.shape[1], np.nan)

    shap_sample = X.sample(min(len(X), 5000), random_state=RANDOM_SEED)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(shap_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    shap_values = np.asarray(shap_values)
    shap_mean_abs = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame(
        {
            "feature": X.columns,
            "gain_importance": [float(gain_importance.get(feature, 0.0)) for feature in X.columns],
            "weight_importance": [float(weight_importance.get(feature, 0.0)) for feature in X.columns],
            "permutation_mean": permutation_mean,
            "permutation_std": permutation_std,
            "shap_mean_abs": shap_mean_abs,
        }
    ).sort_values("shap_mean_abs", ascending=False)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(output_csv, index=False)

    if shap_output_png is not None:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, shap_sample, show=False)
        plt.tight_layout()
        shap_output_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(shap_output_png, dpi=300, bbox_inches="tight")
        plt.close()

        # reproducibility export added: save SHAP values and the sampled X used to compute them,
        # so the notebooks can rebuild the beeswarm without re-running the model pipeline.
        shap_values_npy = shap_output_png.with_name(shap_output_png.stem + "_values.npy")
        shap_sample_csv = shap_output_png.with_name(shap_output_png.stem + "_sample_X.csv")
        np.save(shap_values_npy, shap_values)
        shap_sample.to_csv(shap_sample_csv, index=False)

    return importance_df


def predict_probability_raster(
    model: XGBClassifier,
    feature_names: list[str],
    predictor_arrays: dict[str, np.ndarray],
    reference: ReferenceGrid,
    output_path: Path,
) -> np.ndarray:
    valid_mask = reference.valid_mask.copy()
    for feature_name in feature_names:
        valid_mask &= np.isfinite(predictor_arrays[feature_name])

    prediction = np.full((reference.height, reference.width), PREDICTION_NODATA, dtype=np.float32)
    if valid_mask.any():
        X_grid = np.column_stack([predictor_arrays[feature_name][valid_mask] for feature_name in feature_names])
        prediction[valid_mask] = model.predict_proba(X_grid)[:, 1].astype(np.float32)

    write_array_as_raster(prediction, reference, output_path, dtype="float32", nodata=PREDICTION_NODATA)
    return prediction


def create_model_comparison_table(
    species_key: str,
    env_summary: dict[str, Any],
    conifer_summary: dict[str, Any],
    output_path: Path,
) -> pd.DataFrame:
    comparison_df = pd.DataFrame(
        [
            {
                "model": f"{species_key}_env_only",
                "predictor_set": "environment_only",
                **env_summary,
            },
            {
                "model": f"{species_key}_env_conifer",
                "predictor_set": "environment_plus_conifer_filters",
                **conifer_summary,
            },
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_path, index=False)
    return comparison_df


def top_predictors(importance_df: pd.DataFrame, count: int = 3) -> list[str]:
    ranked = importance_df.sort_values("shap_mean_abs", ascending=False)["feature"].tolist()
    padded = ranked[:count] + [""] * max(0, count - len(ranked[:count]))
    return padded[:count]


def conifer_importance_summary(importance_df: pd.DataFrame) -> str:
    conifer_df = importance_df[importance_df["feature"].isin(CONIFER_KERNEL_PREDICTORS + CONIFER_DISTANCE_PREDICTORS)].copy()
    if conifer_df.empty:
        return "Not applicable"
    conifer_df = conifer_df.sort_values("shap_mean_abs", ascending=False)
    return "; ".join(
        f"{row.feature}={row.shap_mean_abs:.4f}" for row in conifer_df.itertuples()
    )


def run_species_analysis(
    species_key: str,
    paths: ProjectPaths,
    allow_conifer_creation: bool = True,
) -> dict[str, Any]:
    ensure_output_directories(paths)
    environment_stack_report = ensure_projected_environment_stack(paths)
    config = SPECIES_CONFIG[species_key]
    outputs_dir = getattr(paths, config["outputs_dir"])
    response_dir = getattr(paths, config["response_dir"])

    reference = load_reference_grid(paths.reference_raster)
    environment_report = validate_environment_rasters(paths, reference, strict=True)

    occurrence_csv = getattr(paths, config["occurrence_csv"])
    points, point_report = load_occurrence_points(occurrence_csv, reference)

    response_raster_path = response_dir / config["response_raster"]
    response_report = rasterize_response(points, reference, response_raster_path)

    conifer_paths, conifer_report = ensure_conifer_covariates(
        paths,
        reference,
        allow_create=allow_conifer_creation,
    )

    env_df, conifer_df, training_report = build_training_tables(
        species_key,
        paths,
        reference,
        response_raster_path,
        conifer_paths,
    )

    response_column = config["response_column"]
    env_model, env_fold_metrics, env_summary = evaluate_model(
        env_df,
        ENVIRONMENTAL_PREDICTORS,
        response_column,
        paths.models_dir / config["env_model"],
    )
    conifer_model, conifer_fold_metrics, conifer_summary = evaluate_model(
        conifer_df,
        MODEL_B_PREDICTORS,
        response_column,
        paths.models_dir / config["conifer_model"],
    )

    comparison_df = create_model_comparison_table(
        species_key,
        env_summary,
        conifer_summary,
        outputs_dir / config["comparison_csv"],
    )

    # reproducibility export added: write the artefacts needed by Supplementary 5/6
    # to regenerate figures (SHAP, feature importance, model performance) directly
    # from a notebook, without re-running the full pipeline.
    repro_models_dir = outputs_dir / "models"
    repro_models_dir.mkdir(parents=True, exist_ok=True)
    with open(repro_models_dir / f"{species_key}_env_only_model.pkl", "wb") as handle:
        pickle.dump({"model": env_model, "feature_names": ENVIRONMENTAL_PREDICTORS}, handle)
    with open(repro_models_dir / f"{species_key}_env_conifer_model.pkl", "wb") as handle:
        pickle.dump({"model": conifer_model, "feature_names": MODEL_B_PREDICTORS}, handle)

    conifer_df.to_csv(outputs_dir / f"training_table_{species_key}.csv", index=False)
    env_df.to_csv(outputs_dir / f"training_table_{species_key}_env_only.csv", index=False)

    with open(outputs_dir / f"model_b_predictors_{species_key}.txt", "w", encoding="utf-8") as handle:
        handle.write("\n".join(MODEL_B_PREDICTORS) + "\n")
    with open(outputs_dir / f"model_a_predictors_{species_key}.txt", "w", encoding="utf-8") as handle:
        handle.write("\n".join(ENVIRONMENTAL_PREDICTORS) + "\n")

    env_arrays, _ = assemble_predictor_arrays(environmental_raster_paths(paths), reference)
    conifer_prediction_paths = {
        predictor: conifer_paths[predictor] for predictor in CONIFER_KERNEL_PREDICTORS + CONIFER_DISTANCE_PREDICTORS
    }
    conifer_arrays, _ = assemble_predictor_arrays(conifer_prediction_paths, reference)
    prediction_arrays = {**env_arrays, **conifer_arrays}

    env_probability = predict_probability_raster(
        env_model,
        ENVIRONMENTAL_PREDICTORS,
        prediction_arrays,
        reference,
        outputs_dir / config["env_probability"],
    )
    conifer_probability = predict_probability_raster(
        conifer_model,
        MODEL_B_PREDICTORS,
        prediction_arrays,
        reference,
        outputs_dir / config["conifer_probability"],
    )

    difference_raster = np.where(
        (env_probability != PREDICTION_NODATA) & (conifer_probability != PREDICTION_NODATA),
        env_probability - conifer_probability,
        PREDICTION_NODATA,
    ).astype(np.float32)
    write_array_as_raster(
        difference_raster,
        reference,
        outputs_dir / config["difference_raster"],
        dtype="float32",
        nodata=PREDICTION_NODATA,
    )

    env_importance = compute_feature_importance(
        env_model,
        env_df[ENVIRONMENTAL_PREDICTORS],
        env_df[response_column],
        outputs_dir / config["env_importance"],
        shap_output_png=None,
    )
    conifer_importance = compute_feature_importance(
        conifer_model,
        conifer_df[MODEL_B_PREDICTORS],
        conifer_df[response_column],
        outputs_dir / config["conifer_importance"],
        shap_output_png=outputs_dir / config["conifer_shap"],
    )

    return {
        "species": species_key,
        "environment_stack_report": environment_stack_report,
        "reference": {
            "crs": str(reference.crs),
            "transform": tuple(reference.transform),
            "width": reference.width,
            "height": reference.height,
            "resolution": reference.resolution,
            "nodata": reference.nodata,
            "valid_cells": int(reference.valid_mask.sum()),
        },
        "environment_report": environment_report,
        "point_report": point_report,
        "response_report": response_report,
        "conifer_report": conifer_report,
        "training_report": training_report,
        "env_fold_metrics": env_fold_metrics,
        "conifer_fold_metrics": conifer_fold_metrics,
        "comparison_df": comparison_df,
        "env_importance": env_importance,
        "conifer_importance": conifer_importance,
    }


def build_cross_species_comparison(paths: ProjectPaths, species_results: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in species_results:
        species = result["species"]
        comparison_df = result["comparison_df"]
        env_importance = result["env_importance"]
        conifer_importance = result["conifer_importance"]

        for row in comparison_df.itertuples():
            if row.predictor_set == "environment_only":
                predictors = top_predictors(env_importance)
                conifer_summary = "Not applicable"
            else:
                predictors = top_predictors(conifer_importance)
                conifer_summary = conifer_importance_summary(conifer_importance)

            rows.append(
                {
                    "species": species,
                    "model": row.model,
                    "predictor_set": row.predictor_set,
                    "mean_ROC_AUC": row.mean_ROC_AUC,
                    "sd_ROC_AUC": row.sd_ROC_AUC,
                    "mean_PR_AUC": row.mean_PR_AUC,
                    "sd_PR_AUC": row.sd_PR_AUC,
                    "mean_sensitivity": row.mean_sensitivity,
                    "mean_specificity": row.mean_specificity,
                    "mean_TSS": row.mean_TSS,
                    "top_predictor_1": predictors[0],
                    "top_predictor_2": predictors[1],
                    "top_predictor_3": predictors[2],
                    "conifer_predictor_importance_summary": conifer_summary,
                }
            )

    comparison_output = paths.outputs_dir / "acer_cross_species_conifer_comparison.csv"
    cross_species_df = pd.DataFrame(rows)
    comparison_output.parent.mkdir(parents=True, exist_ok=True)
    cross_species_df.to_csv(comparison_output, index=False)
    return cross_species_df


def validate_species_inputs(species_key: str, paths: ProjectPaths) -> dict[str, Any]:
    config = SPECIES_CONFIG[species_key]
    environment_stack_report = ensure_projected_environment_stack(paths)
    reference = load_reference_grid(paths.reference_raster)
    report = {
        "species": species_key,
        "environment_stack_report": environment_stack_report,
        "reference": {
            "crs": str(reference.crs),
            "transform": tuple(reference.transform),
            "width": reference.width,
            "height": reference.height,
            "resolution": reference.resolution,
            "nodata": reference.nodata,
        },
        "environment": validate_environment_rasters(paths, reference, strict=False),
    }

    occurrence_csv = getattr(paths, config["occurrence_csv"])
    occurrence_columns = pd.read_csv(occurrence_csv, nrows=0).columns.tolist()
    forestry_columns = pd.read_csv(paths.forestry_csv, nrows=0).columns.tolist()
    report["occurrence_columns"] = occurrence_columns
    report["forestry_columns"] = forestry_columns

    issues: list[str] = []
    if report["environment"]["missing_rasters"]:
        issues.append("missing environmental rasters")
    if report["environment"]["mismatched_rasters"]:
        issues.append("environment raster alignment or CRS mismatch")
    if reference.crs is None:
        issues.append("reference raster has no CRS")
    elif reference.crs.is_geographic:
        issues.append("reference raster is geographic so kilometre distance rasters are unsafe")

    report["issues"] = issues
    if issues:
        raise DataValidationError("; ".join(issues))

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid-based Acer distribution modelling with conifer-associated ecological filters."
    )
    parser.add_argument(
        "--project-dir",
        default=r"D:\Maple\Articule_Maple",
        help="Base project directory.",
    )
    parser.add_argument(
        "--species",
        choices=["sugar", "red", "both"],
        default="both",
        help="Species analysis to run.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run the input validation checks without fitting models or writing outputs beyond required directories.",
    )
    args = parser.parse_args()

    paths = ProjectPaths.from_project_dir(args.project_dir)
    ensure_output_directories(paths)

    species_to_run = ["sugar", "red"] if args.species == "both" else [args.species]

    if args.check_only:
        for species_key in species_to_run:
            report = validate_species_inputs(species_key, paths)
            print(f"Validation passed for {species_key}: {report}")
        return

    species_results = [run_species_analysis(species_key, paths) for species_key in species_to_run]
    if len(species_results) == 2:
        cross_species = build_cross_species_comparison(paths, species_results)
        print(cross_species)


if __name__ == "__main__":
    main()