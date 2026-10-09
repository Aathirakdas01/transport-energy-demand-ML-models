from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, ParameterSampler


# =============================================================================
# 1) FILES AND GENERAL SETTINGS
# =============================================================================

REPOSITORY_ROOT = Path(__file__).resolve().parent

DEFAULT_DATA_PATH = REPOSITORY_ROOT / "data" / "transport-energy-dataset.csv"
DEFAULT_SHAPEFILE_PATH = REPOSITORY_ROOT / "data" / "SMALL_AREA_2022.shp"
DEFAULT_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "outputs"

DATA_SA_KEY = "area_id"
SHAPEFILE_SA_KEY = "SA_GUID_21"
PROJECTED_CRS = "EPSG:2157"

RANDOM_SEED = 42
N_SPATIAL_FOLDS = 5
GRID_SIZE_M = 30_000.0

N_SEARCH_ITERATIONS = 30
MAX_BOOSTING_ROUNDS = 2_000
EARLY_STOPPING_ROUNDS = 80

XGBOOST_PARAMETER_SPACE = {
    "learning_rate": [0.01, 0.02, 0.03, 0.05],
    "max_depth": [3, 4, 5, 6, 8],
    "min_child_weight": [1, 3, 5, 10],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "gamma": [0.0, 0.1, 0.25, 0.5],
    "reg_alpha": [0.0, 0.1, 0.5, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
}


# =============================================================================
# 2) MODEL SPECIFICATIONS
# =============================================================================

COMMON_FEATURES = [
    "active_travel_share_pct",
    "work_from_home_share_pct",
    "multi_car_household_share_pct",
    "distance_to_built_up_area_km",
    "ptal_index",
    "pobal_index",
]

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "total": {
        "label": "Total",
        "target": "total_energy_demand_million_mj",
        "features": COMMON_FEATURES
        + [
            "population",
            "road_length_km",
        ],
        "log_features": [],
        "log_y": False,
    },
    "per_capita": {
        "label": "Per-capita",
        "target": "per_capita_energy_demand_mj_per_person",
        "features": COMMON_FEATURES
        + [
            "population_density_persons_per_km2",
            "road_density_km_per_km2",
        ],
        "log_features": [
            "population_density_persons_per_km2",
            "road_density_km_per_km2",
        ],
        "log_y": True,
    },
}


# =============================================================================
# 3) DEPENDENCY AND COMMAND-LINE HELPERS
# =============================================================================

def import_xgboost() -> Any:
    """Import XGBoost only when model fitting is requested."""
    try:
        import xgboost as xgb
    except ImportError as error:
        raise ImportError(
            "XGBoost is not installed. Install the repository dependencies "
            "before running this script."
        ) from error
    return xgb


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit XGBoost models for total and/or per-capita private-car "
            "commuting energy demand using spatial GroupKFold validation."
        )
    )
    parser.add_argument(
        "--model",
        choices=["total", "per_capita", "both"],
        default="both",
        help="Model specification to run. Default: both.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to transport-energy-dataset.csv.",
    )
    parser.add_argument(
        "--shapefile",
        type=Path,
        default=DEFAULT_SHAPEFILE_PATH,
        help="Path to SMALL_AREA_2022.shp.",
    )
    parser.add_argument(
        "--sa-key-shp",
        default=SHAPEFILE_SA_KEY,
        help="Small Area identifier column in the shapefile.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for model outputs.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=N_SEARCH_ITERATIONS,
        help="Number of randomized hyperparameter trials.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run 2 tuning trials with 250 boosting rounds for a quick test.",
    )
    return parser.parse_args()


# =============================================================================
# 4) LOAD DATA AND DERIVE EASTING/NORTHING FROM SMALL AREA GEOMETRY
# =============================================================================

def load_dataset(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    data = pd.read_csv(data_path)

    if DATA_SA_KEY not in data.columns:
        raise ValueError(
            f"'{DATA_SA_KEY}' is not present in {data_path.name}."
        )

    data[DATA_SA_KEY] = data[DATA_SA_KEY].astype(str).str.strip()

    if data[DATA_SA_KEY].eq("").any():
        raise ValueError(f"{DATA_SA_KEY} contains empty identifiers.")

    if data[DATA_SA_KEY].duplicated().any():
        n_duplicates = int(data[DATA_SA_KEY].duplicated().sum())
        raise ValueError(
            f"{DATA_SA_KEY} must be unique; {n_duplicates} duplicate(s) found."
        )

    return data


def derive_small_area_coordinates(
    shapefile_path: Path,
    shapefile_sa_key: str,
) -> pd.DataFrame:
    """
    Read the Small Area shapefile, project to EPSG:2157, and calculate
    centroid Easting/Northing coordinates in metres.
    """
    if not shapefile_path.exists():
        raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")

    sa_gdf = gpd.read_file(shapefile_path)

    if shapefile_sa_key not in sa_gdf.columns:
        raise ValueError(
            f"'{shapefile_sa_key}' is not present in the shapefile. "
            f"Available columns include: {list(sa_gdf.columns)}"
        )

    if sa_gdf.crs is None:
        raise ValueError("The Small Area shapefile has no CRS information.")

    sa_gdf = sa_gdf.to_crs(PROJECTED_CRS)
    sa_gdf[shapefile_sa_key] = (
        sa_gdf[shapefile_sa_key].astype(str).str.strip()
    )

    sa_gdf = sa_gdf[[shapefile_sa_key, "geometry"]].copy()
    centroids = sa_gdf.geometry.centroid

    sa_gdf["Easting"] = centroids.x
    sa_gdf["Northing"] = centroids.y

    sa_xy = (
        sa_gdf[[shapefile_sa_key, "Easting", "Northing"]]
        .drop_duplicates(subset=[shapefile_sa_key])
        .copy()
    )

    print("Unique Small Areas in shapefile:", sa_xy[shapefile_sa_key].nunique())
    return sa_xy


def merge_coordinates(
    data: pd.DataFrame,
    sa_xy: pd.DataFrame,
    shapefile_sa_key: str,
) -> pd.DataFrame:
    merged = data.merge(
        sa_xy,
        left_on=DATA_SA_KEY,
        right_on=shapefile_sa_key,
        how="left",
    )

    missing_mask = merged[["Easting", "Northing"]].isna().any(axis=1)
    missing_ids = merged.loc[missing_mask, DATA_SA_KEY].tolist()

    print(f"Rows before coordinate matching: {len(merged):,}")
    print(f"Rows without a centroid match: {len(missing_ids):,}")

    if missing_ids:
        print("Examples of unmatched Small Area IDs:", missing_ids[:10])

    merged = merged.loc[~missing_mask].copy()

    if shapefile_sa_key != DATA_SA_KEY:
        merged = merged.drop(columns=[shapefile_sa_key])

    print(f"Rows retained after coordinate matching: {len(merged):,}")
    return merged


# =============================================================================
# 5) PREPARE TOTAL OR PER-CAPITA MODEL DATA
# =============================================================================

def prepare_model_data(
    merged_data: pd.DataFrame,
    model_name: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    spec = MODEL_SPECS[model_name]
    target_column = spec["target"]
    feature_columns = spec["features"]
    log_feature_columns = spec["log_features"]
    log_y = bool(spec["log_y"])

    required = [
        DATA_SA_KEY,
        target_column,
        *feature_columns,
        "Easting",
        "Northing",
    ]
    missing = sorted(set(required) - set(merged_data.columns))
    if missing:
        raise ValueError(
            f"{spec['label']} model is missing required column(s): "
            + ", ".join(missing)
        )

    model_data = merged_data[required].copy()

    numeric_columns = [
        target_column,
        *feature_columns,
        "Easting",
        "Northing",
    ]
    model_data[numeric_columns] = model_data[numeric_columns].apply(
        pd.to_numeric,
        errors="raise",
    )

    columns_with_missing_values = (
        model_data.columns[model_data.isna().any()].tolist()
    )
    if columns_with_missing_values:
        raise ValueError(
            f"{spec['label']} model contains missing values in: "
            + ", ".join(columns_with_missing_values)
        )

    numeric_array = model_data[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric_array).all():
        raise ValueError(
            f"{spec['label']} model contains infinite numeric values."
        )

    X = model_data[feature_columns].astype(float).copy()
    y = model_data[target_column].astype(float).copy()
    coords = model_data[["Easting", "Northing"]].astype(float).copy()
    area_ids = model_data[DATA_SA_KEY].astype(str).copy()

    for column in log_feature_columns:
        if (X[column] < -1).any():
            raise ValueError(
                f"Cannot apply log(x + 1) to '{column}' because values < -1 exist."
            )
        X[column] = np.log(X[column] + 1.0)

    if log_y:
        if (y < -1).any():
            raise ValueError(
                f"Cannot apply log(y + 1) to target '{target_column}'."
            )
        y = np.log(y + 1.0)

    print(f"\n--- {spec['label']} XGBoost model ---")
    print("Observations:", len(X))
    print("Target:", target_column)
    print("Features:", feature_columns)
    
    return X, y, coords, area_ids


# =============================================================================
# 6) SPATIAL BLOCKING BY REGULAR GRID
# =============================================================================

def make_spatial_groups(
    coords: pd.DataFrame,
    label: str,
    grid_size_m: float = GRID_SIZE_M,
    n_splits: int = N_SPATIAL_FOLDS,
    random_state: int = RANDOM_SEED,
) -> tuple[np.ndarray, GroupKFold]:
    if coords[["Easting", "Northing"]].isna().any().any():
        raise ValueError(f"{label}: Some rows are missing Easting/Northing.")

    xmin = float(coords["Easting"].min())
    ymin = float(coords["Northing"].min())

    x_bins = np.floor(
        (coords["Easting"] - xmin) / grid_size_m
    ).astype(int)
    y_bins = np.floor(
        (coords["Northing"] - ymin) / grid_size_m
    ).astype(int)

    grid_id = x_bins.astype(str) + "_" + y_bins.astype(str)

    unique_cells = grid_id.unique().astype(object)
    if len(unique_cells) < n_splits:
        raise ValueError(
            f"{label}: only {len(unique_cells)} spatial grid cells were "
            f"created; at least {n_splits} are required."
        )

    rng = np.random.RandomState(random_state)
    rng.shuffle(unique_cells)

    cell_to_fold = {
        cell: i % n_splits for i, cell in enumerate(unique_cells)
    }
    groups = grid_id.map(cell_to_fold).to_numpy(dtype=int)

    print(
        f"{label} spatial fold sizes (grid-block CV):",
        pd.Series(groups).value_counts().sort_index().to_dict(),
    )

    return groups, GroupKFold(n_splits=n_splits)


# =============================================================================
# 7) XGBOOST SPATIAL GROUPKFOLD CROSS-VALIDATION
# =============================================================================

def predict_xgb_best_iteration(
    booster: Any,
    dmatrix: Any,
    best_iteration: int,
) -> np.ndarray:
    """
    Predict using trees from iteration 0 to best_iteration - 1.
    Includes a fallback for older XGBoost versions.
    """
    try:
        return booster.predict(
            dmatrix,
            iteration_range=(0, int(best_iteration)),
        )
    except TypeError:
        return booster.predict(
            dmatrix,
            ntree_limit=int(best_iteration),
        )


def eval_xgboost_spatial_cv(
    xgb: Any,
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    gkf: GroupKFold,
    seed: int = RANDOM_SEED,
    xgb_params: dict[str, Any] | None = None,
    num_boost_round: int = MAX_BOOSTING_ROUNDS,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    verbose_eval: bool = False,
    return_oof: bool = False,
) -> dict[str, Any]:
    if xgb_params is None:
        xgb_params = {}

    if len(X) != len(y):
        raise ValueError(f"X and y lengths differ: {len(X)} vs {len(y)}")
    if len(groups) != len(X):
        raise ValueError(
            f"groups and X lengths differ: {len(groups)} vs {len(X)}"
        )
    if X.isna().any().any():
        raise ValueError("X contains missing values.")
    if y.isna().any():
        raise ValueError("y contains missing values.")

    X_np = X.to_numpy(dtype=float)
    y_np = y.to_numpy(dtype=float)
    feature_names = [str(column) for column in X.columns]

    fold_r2: list[float] = []
    fold_rmse: list[float] = []
    fold_mae: list[float] = []
    fold_mse: list[float] = []
    best_iters: list[int] = []

    oof_predictions = np.full(len(y_np), np.nan, dtype=float)
    oof_fold = np.full(len(y_np), -1, dtype=int)

    for fold_id, (tr_idx, va_idx) in enumerate(
        gkf.split(X_np, y_np, groups=groups),
        start=1,
    ):
        X_tr, X_va = X_np[tr_idx], X_np[va_idx]
        y_tr, y_va = y_np[tr_idx], y_np[va_idx]

        dtrain = xgb.DMatrix(
            X_tr,
            label=y_tr,
            feature_names=feature_names,
        )
        dvalid = xgb.DMatrix(
            X_va,
            label=y_va,
            feature_names=feature_names,
        )

        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "seed": int(seed),
            "nthread": -1,
        }
        params.update(xgb_params)

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=int(num_boost_round),
            evals=[(dvalid, "validation")],
            early_stopping_rounds=int(early_stopping_rounds),
            verbose_eval=verbose_eval,
        )

        best_iteration = int(booster.best_iteration) + 1

        pred = predict_xgb_best_iteration(
            booster=booster,
            dmatrix=dvalid,
            best_iteration=best_iteration,
        )

        mse_val = float(mean_squared_error(y_va, pred))
        rmse_val = float(np.sqrt(mse_val))
        mae_val = float(mean_absolute_error(y_va, pred))
        r2_val = float(r2_score(y_va, pred))

        fold_mse.append(mse_val)
        fold_rmse.append(rmse_val)
        fold_mae.append(mae_val)
        fold_r2.append(r2_val)
        best_iters.append(best_iteration)

        oof_predictions[va_idx] = pred
        oof_fold[va_idx] = fold_id

        print(
            f"Fold {fold_id}: "
            f"R2={r2_val:.4f} | "
            f"RMSE={rmse_val:.4f} | "
            f"MAE={mae_val:.4f} | "
            f"MSE={mse_val:.4f} | "
            f"best_iter={best_iteration}"
        )

    results: dict[str, Any] = {
        "mean_r2": float(np.mean(fold_r2)),
        "std_r2": float(np.std(fold_r2)),
        "mean_rmse": float(np.mean(fold_rmse)),
        "std_rmse": float(np.std(fold_rmse)),
        "mean_mae": float(np.mean(fold_mae)),
        "mean_mse": float(np.mean(fold_mse)),
        "mean_best_iter": float(np.mean(best_iters)),
        "fold_r2": fold_r2,
        "fold_rmse": fold_rmse,
        "fold_mae": fold_mae,
        "fold_mse": fold_mse,
        "best_iters": best_iters,
    }

    if return_oof:
        if np.isnan(oof_predictions).any():
            raise ValueError(
                "Some observations did not receive an OOF prediction."
            )
        if (oof_fold < 1).any():
            raise ValueError(
                "Some observations were not assigned to a validation fold."
            )

        results["oof_predictions"] = oof_predictions
        results["oof_fold"] = oof_fold

    return results


# =============================================================================
# 8) RANDOMIZED HYPERPARAMETER SEARCH
# =============================================================================

def tune_xgboost_spatial(
    xgb: Any,
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    gkf: GroupKFold,
    n_iter: int = N_SEARCH_ITERATIONS,
    seed: int = RANDOM_SEED,
    num_boost_round: int = MAX_BOOSTING_ROUNDS,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
) -> tuple[dict[str, Any], pd.DataFrame]:
    sampler = list(
        ParameterSampler(
            XGBOOST_PARAMETER_SPACE,
            n_iter=n_iter,
            random_state=seed,
        )
    )

    best: dict[str, Any] | None = None
    results: list[dict[str, Any]] = []

    for trial, trial_params in enumerate(sampler, start=1):
        print("\n" + "=" * 70)
        print(f"XGBoost trial {trial}/{n_iter}")
        print(trial_params)
        print("=" * 70)

        stats = eval_xgboost_spatial_cv(
            xgb=xgb,
            X=X,
            y=y,
            groups=groups,
            gkf=gkf,
            seed=seed,
            xgb_params=trial_params,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
            return_oof=False,
        )

        row = {"trial": trial, **trial_params, **stats}
        results.append(row)

        if best is None or stats["mean_r2"] > best["mean_r2"]:
            best = row
            print(
                f"\n[Trial {trial:03d}/{n_iter}] "
                f"New best R2={best['mean_r2']:.4f} | "
                f"RMSE={best['mean_rmse']:.4f} | "
                f"MAE={best['mean_mae']:.4f} | "
                f"best_iter≈{best['mean_best_iter']:.0f}"
            )

    if best is None:
        raise RuntimeError("No XGBoost hyperparameter trial was evaluated.")

    results_df = (
        pd.DataFrame(results)
        .sort_values("mean_r2", ascending=False)
        .reset_index(drop=True)
    )

    return best, results_df


# =============================================================================
# 9) SAVE OOF RESULTS AND FIT FINAL MODEL
# =============================================================================

def original_scale(values: np.ndarray, log_y: bool) -> np.ndarray:
    return np.expm1(values) if log_y else values.copy()


def save_model_outputs(
    xgb: Any,
    merged_data: pd.DataFrame,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    coords: pd.DataFrame,
    area_ids: pd.Series,
    groups: np.ndarray,
    gkf: GroupKFold,
    best: dict[str, Any],
    tuning_results: pd.DataFrame,
    output_directory: Path,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> None:
    spec = MODEL_SPECS[model_name]
    prefix = f"xgboost_{model_name}"

    model_output_dir = output_directory / f"XGBoost_{model_name}"
    model_output_dir.mkdir(parents=True, exist_ok=True)

    tuning_results.to_csv(
        model_output_dir / f"{prefix}_tuning_results.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "feature_position": np.arange(1, len(X.columns) + 1),
            "feature_name": X.columns,
        }
    ).to_csv(
        model_output_dir / f"{prefix}_feature_order.csv",
        index=False,
    )

    best_params = {
        key: best[key] for key in XGBOOST_PARAMETER_SPACE.keys()
    }
    best_number_of_rounds = max(
        1,
        int(round(best["mean_best_iter"])),
    )

    pd.DataFrame(
        [
            {
                **best_params,
                "num_boost_round": best_number_of_rounds,
                "seed": RANDOM_SEED,
                "number_of_spatial_folds": N_SPATIAL_FOLDS,
                "mean_cv_r2": best["mean_r2"],
                "std_cv_r2": best["std_r2"],
                "mean_cv_rmse": best["mean_rmse"],
                "std_cv_rmse": best["std_rmse"],
                "mean_cv_mae": best["mean_mae"],
                "mean_cv_mse": best["mean_mse"],
            }
        ]
    ).to_csv(
        model_output_dir / f"{prefix}_best_parameters.csv",
        index=False,
    )

    print(f"\nGenerating {spec['label']} out-of-fold predictions...")

    best_oof = eval_xgboost_spatial_cv(
        xgb=xgb,
        X=X,
        y=y,
        groups=groups,
        gkf=gkf,
        seed=RANDOM_SEED,
        xgb_params=best_params,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
        return_oof=True,
    )

    y_model_observed = y.to_numpy(dtype=float)
    y_model_predicted = best_oof["oof_predictions"]

    log_y = bool(spec["log_y"])
    y_original_observed = original_scale(y_model_observed, log_y)
    y_original_predicted = original_scale(y_model_predicted, log_y)

    residual_model = y_model_observed - y_model_predicted
    residual_original = y_original_observed - y_original_predicted

    model_mse = float(
        mean_squared_error(y_model_observed, y_model_predicted)
    )
    original_mse = float(
        mean_squared_error(y_original_observed, y_original_predicted)
    )

    metrics = pd.DataFrame(
        [
            {
                "model": "XGBoost",
                "outcome": spec["label"],
                "scale": "modelled_log_scale" if log_y else "modelled_scale",
                "r2": r2_score(y_model_observed, y_model_predicted),
                "rmse": np.sqrt(model_mse),
                "mae": mean_absolute_error(
                    y_model_observed, y_model_predicted
                ),
                "mse": model_mse,
            },
            {
                "model": "XGBoost",
                "outcome": spec["label"],
                "scale": "original_scale",
                "r2": r2_score(y_original_observed, y_original_predicted),
                "rmse": np.sqrt(original_mse),
                "mae": mean_absolute_error(
                    y_original_observed, y_original_predicted
                ),
                "mse": original_mse,
            },
        ]
    )
    metrics.to_csv(
        model_output_dir / f"{prefix}_oof_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "area_id": area_ids.reset_index(drop=True).to_numpy(),
            "Easting": coords["Easting"].reset_index(drop=True).to_numpy(),
            "Northing": coords["Northing"].reset_index(drop=True).to_numpy(),
            "spatial_fold": best_oof["oof_fold"],
            "observed_model_scale": y_model_observed,
            "predicted_oof_model_scale": y_model_predicted,
            "residual_oof_model_scale": residual_model,
            "observed_original_scale": y_original_observed,
            "predicted_oof_original_scale": y_original_predicted,
            "residual_oof_original_scale": residual_original,
            "absolute_residual_original_scale": np.abs(residual_original),
            "squared_residual_original_scale": residual_original ** 2,
        }
    ).to_csv(
        model_output_dir / f"{prefix}_oof_residuals.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "spatial_fold": np.arange(
                1,
                len(best_oof["fold_r2"]) + 1,
            ),
            "r2": best_oof["fold_r2"],
            "rmse": best_oof["fold_rmse"],
            "mae": best_oof["fold_mae"],
            "mse": best_oof["fold_mse"],
            "best_iteration": best_oof["best_iters"],
        }
    ).to_csv(
        model_output_dir / f"{prefix}_fold_metrics.csv",
        index=False,
    )

    feature_names = [str(column) for column in X.columns]
    dtrain_full = xgb.DMatrix(
        X.to_numpy(dtype=float),
        label=y.to_numpy(dtype=float),
        feature_names=feature_names,
    )

    final_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "seed": RANDOM_SEED,
        "nthread": -1,
    }
    final_params.update(best_params)

    final_model = xgb.train(
        params=final_params,
        dtrain=dtrain_full,
        num_boost_round=best_number_of_rounds,
    )

    final_model.save_model(
        str(model_output_dir / f"{prefix}_best_model.json")
    )

    print(f"\nSaved {spec['label']} outputs to: {model_output_dir.resolve()}")


# =============================================================================
# 10) RUN ONE MODEL SPECIFICATION
# =============================================================================

def run_model(
    xgb: Any,
    merged_data: pd.DataFrame,
    model_name: str,
    output_directory: Path,
    n_iter: int,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> None:
    spec = MODEL_SPECS[model_name]

    X, y, coords, area_ids = prepare_model_data(
        merged_data=merged_data,
        model_name=model_name,
    )

    groups, gkf = make_spatial_groups(
        coords=coords,
        label=spec["label"],
    )

    best, tuning_results = tune_xgboost_spatial(
        xgb=xgb,
        X=X,
        y=y,
        groups=groups,
        gkf=gkf,
        n_iter=n_iter,
        seed=RANDOM_SEED,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
    )

    save_model_outputs(
        xgb=xgb,
        merged_data=merged_data,
        model_name=model_name,
        X=X,
        y=y,
        coords=coords,
        area_ids=area_ids,
        groups=groups,
        gkf=gkf,
        best=best,
        tuning_results=tuning_results,
        output_directory=output_directory,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
    )


# =============================================================================
# 11) MAIN WORKFLOW
# =============================================================================

def main() -> None:
    args = parse_arguments()

    data = load_dataset(args.data)
    sa_xy = derive_small_area_coordinates(
        shapefile_path=args.shapefile,
        shapefile_sa_key=args.sa_key_shp,
    )
    merged_data = merge_coordinates(
        data=data,
        sa_xy=sa_xy,
        shapefile_sa_key=args.sa_key_shp,
    )

    n_iter = 2 if args.quick else args.n_iter
    num_boost_round = 250 if args.quick else MAX_BOOSTING_ROUNDS
    early_stopping_rounds = (
        25 if args.quick else EARLY_STOPPING_ROUNDS
    )

    xgb = import_xgboost()

    if args.model == "both":
        models_to_run = ["total", "per_capita"]
    else:
        models_to_run = [args.model]

    for model_name in models_to_run:
        run_model(
            xgb=xgb,
            merged_data=merged_data,
            model_name=model_name,
            output_directory=args.output_dir,
            n_iter=n_iter,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
        )


if __name__ == "__main__":
    main()
