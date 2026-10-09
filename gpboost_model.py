from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


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

COVARIANCE_FUNCTION = "exponential"
MAX_GP_POINTS = 4_000
VECCHIA_NEIGHBORS = 30

BASE_GPBOOST_PARAMETERS = {
    "objective": "regression_l2",
    "metric": "l2",
    "bagging_freq": 0,
    "bagging_fraction": 1.0,
    "verbose": -1,
}

GPBOOST_PARAMETER_SPACE = {
    "learning_rate": [0.01, 0.02, 0.03, 0.05],
    "num_leaves": [31, 63, 127],
    "min_data_in_leaf": [10, 20, 40, 80],
    "feature_fraction": [0.6, 0.7, 0.8, 0.9, 1.0],
    "lambda_l2": [0.0, 0.5, 1.0, 2.0, 5.0],
    "max_depth": [-1, 6, 10],
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

def import_gpboost() -> Any:
    """Import GPBoost only when model fitting is requested."""
    try:
        import gpboost as gpb
    except ImportError as error:
        raise ImportError(
            "GPBoost is not installed. Install the repository dependencies "
            "before running this script."
        ) from error
    return gpb


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit GPBoost models for total and/or per-capita private-car "
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
        help="Number of random hyperparameter trials.",
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

    # Predictor transformation used in the study.
    for column in log_feature_columns:
        if (X[column] < -1).any():
            raise ValueError(
                f"Cannot apply log(x + 1) to '{column}' because values < -1 exist."
            )
        X[column] = np.log(X[column] + 1.0)

    # Response transformation: per-capita only.
    if log_y:
        if (y < -1).any():
            raise ValueError(
                f"Cannot apply log(y + 1) to target '{target_column}'."
            )
        y = np.log(y + 1.0)

    print(f"\n--- {spec['label']} GPBoost model ---")
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
# 7) GPBOOST SPATIAL GROUPKFOLD CROSS-VALIDATION
# =============================================================================

def extract_prediction_mean(prediction_output: Any) -> np.ndarray:
    if isinstance(prediction_output, dict):
        prediction = prediction_output.get(
            "response_mean",
            prediction_output.get("mu"),
        )
        if prediction is None:
            raise ValueError(
                "Unexpected GPBoost prediction dictionary keys: "
                f"{list(prediction_output.keys())}"
            )
    else:
        prediction = prediction_output

    prediction = np.asarray(prediction, dtype=float).reshape(-1)

    if not np.isfinite(prediction).all():
        raise ValueError("GPBoost returned non-finite predictions.")

    return prediction


def build_gp_model(
    gpb: Any,
    training_coordinates: np.ndarray,
    covariance_function: str,
    max_gp_points: int | None,
    vecchia_neighbors: int,
) -> Any:
    gp_arguments: dict[str, Any] = {
        "gp_coords": training_coordinates,
        "cov_function": covariance_function,
    }

    if (
        max_gp_points is not None
        and len(training_coordinates) > max_gp_points
    ):
        gp_arguments.update(
            {
                "gp_approx": "vecchia",
                "num_neighbors": int(vecchia_neighbors),
            }
        )

    return gpb.GPModel(**gp_arguments)


def eval_gpboost_spatial_cv(
    gpb: Any,
    X: pd.DataFrame,
    y: pd.Series,
    coords: pd.DataFrame,
    groups: np.ndarray,
    gkf: GroupKFold,
    seed: int = RANDOM_SEED,
    lgb_params: dict[str, Any] | None = None,
    cov_function: str = COVARIANCE_FUNCTION,
    num_boost_round: int = MAX_BOOSTING_ROUNDS,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    verbose_eval: bool = False,
    max_gp_points: int | None = MAX_GP_POINTS,
    vecchia_neighbors: int = VECCHIA_NEIGHBORS,
    return_oof: bool = False,
) -> dict[str, Any]:
    if lgb_params is None:
        lgb_params = {}

    if len(X) != len(y):
        raise ValueError(f"X and y lengths differ: {len(X)} vs {len(y)}")
    if len(coords) != len(X):
        raise ValueError(
            f"coords and X lengths differ: {len(coords)} vs {len(X)}"
        )
    if len(groups) != len(X):
        raise ValueError(
            f"groups and X lengths differ: {len(groups)} vs {len(X)}"
        )

    X_np = X.to_numpy(dtype=float)
    y_np = y.to_numpy(dtype=float)
    coords_np = coords[["Easting", "Northing"]].to_numpy(dtype=float)
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
        coords_tr, coords_va = coords_np[tr_idx], coords_np[va_idx]

        gp_model = build_gp_model(
            gpb=gpb,
            training_coordinates=coords_tr,
            covariance_function=cov_function,
            max_gp_points=max_gp_points,
            vecchia_neighbors=vecchia_neighbors,
        )
        gp_model.set_prediction_data(gp_coords_pred=coords_va)

        dtrain = gpb.Dataset(
            X_tr,
            label=y_tr,
            feature_name=feature_names,
            free_raw_data=False,
        )
        dvalid = gpb.Dataset(
            X_va,
            label=y_va,
            reference=dtrain,
            feature_name=feature_names,
            free_raw_data=False,
        )

        params = {
            **BASE_GPBOOST_PARAMETERS,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "max_depth": -1,
            "feature_fraction": 0.8,
            "min_data_in_leaf": 20,
            "lambda_l2": 1.0,
            "seed": int(seed),
        }
        params.update(lgb_params)

        booster = gpb.train(
            params=params,
            train_set=dtrain,
            gp_model=gp_model,
            num_boost_round=int(num_boost_round),
            valid_sets=[dvalid],
            valid_names=["validation"],
            early_stopping_rounds=int(early_stopping_rounds),
            verbose_eval=verbose_eval,
        )

        best_iteration = int(booster.best_iteration)
        if best_iteration <= 0:
            best_iteration = int(num_boost_round)

        pred_out = booster.predict(
            data=X_va,
            gp_coords_pred=coords_va,
            num_iteration=best_iteration,
        )
        pred = extract_prediction_mean(pred_out)

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
            raise RuntimeError(
                "Some observations did not receive an out-of-fold prediction."
            )
        if (oof_fold < 1).any():
            raise RuntimeError(
                "Some observations were not assigned to a validation fold."
            )

        results["oof_predictions"] = oof_predictions
        results["oof_fold"] = oof_fold

    return results


# =============================================================================
# 8) RANDOM HYPERPARAMETER SEARCH
# =============================================================================

def make_sampler(
    space: dict[str, list[Any]],
    n_iter: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    samples: list[dict[str, Any]] = []

    for _ in range(n_iter):
        sample: dict[str, Any] = {}
        for key, values in space.items():
            value = rng.choice(values)
            sample[key] = value.item() if hasattr(value, "item") else value
        samples.append(sample)

    return samples


def tune_gpboost_spatial(
    gpb: Any,
    X: pd.DataFrame,
    y: pd.Series,
    coords: pd.DataFrame,
    groups: np.ndarray,
    gkf: GroupKFold,
    n_iter: int = N_SEARCH_ITERATIONS,
    seed: int = RANDOM_SEED,
    cov_function: str = COVARIANCE_FUNCTION,
    num_boost_round: int = MAX_BOOSTING_ROUNDS,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    max_gp_points: int | None = MAX_GP_POINTS,
    vecchia_neighbors: int = VECCHIA_NEIGHBORS,
) -> tuple[dict[str, Any], pd.DataFrame]:
    sampler = make_sampler(
        GPBOOST_PARAMETER_SPACE,
        n_iter=n_iter,
        seed=seed,
    )

    best: dict[str, Any] | None = None
    results: list[dict[str, Any]] = []

    for trial, trial_params in enumerate(sampler, start=1):
        print("\n" + "=" * 70)
        print(f"GPBoost trial {trial}/{n_iter}")
        print(trial_params)
        print("=" * 70)

        stats = eval_gpboost_spatial_cv(
            gpb=gpb,
            X=X,
            y=y,
            coords=coords,
            groups=groups,
            gkf=gkf,
            seed=seed,
            lgb_params=trial_params,
            cov_function=cov_function,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
            max_gp_points=max_gp_points,
            vecchia_neighbors=vecchia_neighbors,
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
        raise RuntimeError("No GPBoost hyperparameter trial was evaluated.")

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
    gpb: Any,
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
    prefix = f"gpboost_{model_name}"

    model_output_dir = output_directory / f"GPBoost_{model_name}"
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # 1. All tuning trials
    tuning_results.to_csv(
        model_output_dir / f"{prefix}_tuning_results.csv",
        index=False,
    )

    # 2. Feature order used by the model
    pd.DataFrame(
        {
            "feature_position": np.arange(1, len(X.columns) + 1),
            "feature_name": X.columns,
        }
    ).to_csv(
        model_output_dir / f"{prefix}_feature_order.csv",
        index=False,
    )

    # 3. Selected hyperparameters and summary CV performance
    best_params = {
        key: best[key] for key in GPBOOST_PARAMETER_SPACE.keys()
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
                "covariance_function": COVARIANCE_FUNCTION,
                "max_gp_points": MAX_GP_POINTS,
                "vecchia_neighbors": VECCHIA_NEIGHBORS,
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

    # Re-run the selected specification to obtain one OOF prediction per SA.
    print(f"\nGenerating {spec['label']} GPBoost out-of-fold predictions...")

    best_oof = eval_gpboost_spatial_cv(
        gpb=gpb,
        X=X,
        y=y,
        coords=coords,
        groups=groups,
        gkf=gkf,
        seed=RANDOM_SEED,
        lgb_params=best_params,
        cov_function=COVARIANCE_FUNCTION,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
        max_gp_points=MAX_GP_POINTS,
        vecchia_neighbors=VECCHIA_NEIGHBORS,
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

    # 4. Overall OOF metrics on modelled and original scales
    pd.DataFrame(
        [
            {
                "model": "GPBoost",
                "outcome": spec["label"],
                "scale": "modelled_log_scale" if log_y else "modelled_scale",
                "r2": r2_score(y_model_observed, y_model_predicted),
                "rmse": np.sqrt(model_mse),
                "mae": mean_absolute_error(
                    y_model_observed,
                    y_model_predicted,
                ),
                "mse": model_mse,
            },
            {
                "model": "GPBoost",
                "outcome": spec["label"],
                "scale": "original_scale",
                "r2": r2_score(
                    y_original_observed,
                    y_original_predicted,
                ),
                "rmse": np.sqrt(original_mse),
                "mae": mean_absolute_error(
                    y_original_observed,
                    y_original_predicted,
                ),
                "mse": original_mse,
            },
        ]
    ).to_csv(
        model_output_dir / f"{prefix}_oof_metrics.csv",
        index=False,
    )

    # 5. SA-level OOF predictions and residuals
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

    # 6. Metrics for each spatial validation fold
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

    # 7. Fit the selected specification to the complete dataset and save it.
    feature_values = X.to_numpy(dtype=float)
    target_values = y.to_numpy(dtype=float)
    coordinate_values = coords[
        ["Easting", "Northing"]
    ].to_numpy(dtype=float)
    feature_names = [str(column) for column in X.columns]

    final_gp_model = build_gp_model(
        gpb=gpb,
        training_coordinates=coordinate_values,
        covariance_function=COVARIANCE_FUNCTION,
        max_gp_points=MAX_GP_POINTS,
        vecchia_neighbors=VECCHIA_NEIGHBORS,
    )

    final_training_data = gpb.Dataset(
        feature_values,
        label=target_values,
        feature_name=feature_names,
        free_raw_data=False,
    )

    final_params = {
        **BASE_GPBOOST_PARAMETERS,
        **best_params,
        "seed": RANDOM_SEED,
    }

    final_model = gpb.train(
        params=final_params,
        train_set=final_training_data,
        gp_model=final_gp_model,
        train_gp_model_cov_pars=True,
        num_boost_round=best_number_of_rounds,
        verbose_eval=False,
    )

    final_model.save_model(
        str(model_output_dir / f"{prefix}_best_model.txt"),
        num_iteration=best_number_of_rounds,
    )

    print(
        f"\nSaved {spec['label']} GPBoost outputs to: "
        f"{model_output_dir.resolve()}"
    )


# =============================================================================
# 10) RUN ONE MODEL SPECIFICATION
# =============================================================================

def run_model(
    gpb: Any,
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

    best, tuning_results = tune_gpboost_spatial(
        gpb=gpb,
        X=X,
        y=y,
        coords=coords,
        groups=groups,
        gkf=gkf,
        n_iter=n_iter,
        seed=RANDOM_SEED,
        cov_function=COVARIANCE_FUNCTION,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        max_gp_points=MAX_GP_POINTS,
        vecchia_neighbors=VECCHIA_NEIGHBORS,
    )

    save_model_outputs(
        gpb=gpb,
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

    gpb = import_gpboost()

    if args.model == "both":
        models_to_run = ["total", "per_capita"]
    else:
        models_to_run = [args.model]

    for model_name in models_to_run:
        run_model(
            gpb=gpb,
            merged_data=merged_data,
            model_name=model_name,
            output_directory=args.output_dir,
            n_iter=n_iter,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
        )


if __name__ == "__main__":
    main()
