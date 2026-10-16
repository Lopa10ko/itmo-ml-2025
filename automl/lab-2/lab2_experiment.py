from __future__ import annotations

import json
import os
import re
import traceback
import warnings
from pathlib import Path
from time import perf_counter

LAB_DIR = Path(__file__).resolve().parent
CACHE_DIR = LAB_DIR / ".cache"
RESULTS_DIR = LAB_DIR / "results"
FIGURES_DIR = LAB_DIR / "figures"
for folder in [CACHE_DIR, RESULTS_DIR, FIGURES_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Both libraries otherwise try to write under the user's home directory.
os.environ.setdefault("OPENML_CACHE_DIR", str(CACHE_DIR / "openml"))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import mlflow
import numpy as np
import openml
import pandas as pd
from mlflow.entities import ViewType
from pymfe.mfe import MFE
from scipy.stats import entropy
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

SEED = 334771
N_FOLDS = 3
TARGET_DATASETS = 300
PIPELINE_VERSION = "3"
EXPERIMENT_NAME = "automl_lab2_meta_features"
TRACKING_URI = f"sqlite:///{LAB_DIR / 'mlflow.db'}"

MIN_INSTANCES = 100
MAX_INSTANCES = 5000
MAX_FEATURES = 100
MAX_CLASSES = 10
MAX_CATEGORICAL_FEATURES = 30
MAX_CATEGORY_LEVELS = 50
MAX_PYMFE_COMPLEXITY_ROWS = 500

ALGORITHMS = {
    "Logistic regression": OneVsRestClassifier(
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            solver="liblinear",
            random_state=SEED,
        ),
        n_jobs=1,
    ),
    "RBF SVM": SVC(C=1.0, gamma="scale", class_weight="balanced"),
    "Random forest": RandomForestClassifier(
        n_estimators=80,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=1,
    ),
}

PYMFE_FEATURES = {
    "general": [
        "attr_to_inst",
        "freq_class",
        "nr_attr",
        "nr_cat",
        "nr_class",
        "nr_inst",
        "nr_num",
    ],
    "statistical": ["cor", "kurtosis", "mean", "sd", "skewness"],
    "complexity": ["f1", "f2", "n2", "t2"],
}

MANUAL_FEATURE_DESCRIPTIONS = {
    "basic__n_instances": ("Basic", "Number of usable rows"),
    "basic__n_features": ("Basic", "Number of non-constant predictors"),
    "basic__n_numeric": ("Basic", "Number of numerical predictors"),
    "basic__n_categorical": ("Basic", "Number of categorical predictors"),
    "basic__n_classes": ("Basic", "Number of target classes"),
    "basic__missing_fraction": ("Basic", "Fraction of missing predictor cells"),
    "basic__feature_instance_ratio": ("Basic", "Predictors divided by rows"),
    "stat__minority_fraction": ("Statistical", "Share of the least frequent class"),
    "stat__normalized_class_entropy": ("Statistical", "Class entropy divided by log(number of classes)"),
    "stat__mean_abs_skewness": ("Statistical", "Mean absolute skewness of numerical predictors"),
    "stat__mean_abs_kurtosis": ("Statistical", "Mean absolute excess kurtosis of numerical predictors"),
    "stat__mean_abs_spearman": ("Statistical", "Mean absolute pairwise Spearman correlation"),
    "stat__mean_outlier_fraction": ("Statistical", "Mean fraction beyond the 1.5 IQR fences"),
    "struct__duplicate_fraction": ("Structural", "Fraction of rows duplicated in predictor space"),
    "struct__mean_unique_fraction": ("Structural", "Mean distinct-value count divided by rows"),
    "struct__mean_category_cardinality": ("Structural", "Mean levels per categorical predictor divided by rows"),
    "struct__pca_first_component": ("Structural", "Variance share of the first standardized numerical PC"),
    "struct__pca_components_95_fraction": ("Structural", "Fraction of numerical PCs needed for 95% variance"),
    "struct__mean_fisher_ratio": ("Structural", "Mean between-class to within-class variance ratio"),
}


class DatasetSkip(RuntimeError):
    pass


def configure_tracking() -> str:
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        artifact_uri = (LAB_DIR / "mlartifacts").resolve().as_uri()
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=artifact_uri)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(EXPERIMENT_NAME)
    return str(experiment_id)


def _safe_float(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def _unique_columns(columns) -> list[str]:
    result, used = [], {}
    for raw in columns:
        name = str(raw)
        count = used.get(name, 0)
        used[name] = count + 1
        result.append(name if count == 0 else f"{name}__{count + 1}")
    return result


def load_openml_task(task_id: int):
    task = openml.tasks.get_task(int(task_id), download_splits=False)
    dataset = task.get_dataset()
    X, y, categorical_indicator, _ = dataset.get_data(
        target=task.target_name, dataset_format="dataframe"
    )

    X = X.copy()
    X.columns = _unique_columns(X.columns)
    categorical = [column for column, flag in zip(X.columns, categorical_indicator) if flag]
    usable_rows = y.notna()
    X = X.loc[usable_rows].reset_index(drop=True)
    y = y.loc[usable_rows].reset_index(drop=True)

    for column in X.columns:
        if column in categorical:
            values = X[column].astype(object)
            X[column] = values.where(pd.notna(values), np.nan)
        else:
            X[column] = pd.to_numeric(X[column], errors="coerce").replace([np.inf, -np.inf], np.nan)

    keep = [column for column in X if X[column].nunique(dropna=True) > 1]
    dropped = [column for column in X if column not in keep]
    X = X[keep]
    categorical = [column for column in categorical if column in keep]
    numeric = [column for column in X if column not in categorical]

    labels, label_names = pd.factorize(y.astype(str), sort=True)
    class_counts = np.bincount(labels)
    if len(label_names) < 2:
        raise DatasetSkip("the target has fewer than two classes")
    if len(label_names) > MAX_CLASSES:
        raise DatasetSkip(f"the target has {len(label_names)} classes")
    if class_counts.min() < N_FOLDS:
        raise DatasetSkip("the smallest class cannot support three folds")
    if len(X.columns) < 2:
        raise DatasetSkip("fewer than two non-constant predictors remain")

    info = {
        "task_id": int(task_id),
        "dataset_id": int(task.dataset_id),
        "dataset_name": str(dataset.name),
        "target_name": str(task.target_name),
        "n_dropped_constant": len(dropped),
        "target_levels": [str(value) for value in label_names],
    }
    return X, labels.astype(int), categorical, numeric, info


def candidate_tasks(refresh: bool = False) -> pd.DataFrame:
    path = RESULTS_DIR / "openml_candidates.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path)

    openml.config.set_console_log_level(40)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        tasks = openml.tasks.list_tasks(
            task_type=openml.tasks.TaskType.SUPERVISED_CLASSIFICATION,
            status="active",
            output_format="dataframe",
        )

    keep = (
        tasks["NumberOfInstances"].between(MIN_INSTANCES, MAX_INSTANCES)
        & tasks["NumberOfFeatures"].between(2, MAX_FEATURES)
        & tasks["NumberOfClasses"].between(2, MAX_CLASSES)
        & tasks["MinorityClassSize"].ge(2 * N_FOLDS)
        & tasks["NumberOfSymbolicFeatures"].le(MAX_CATEGORICAL_FEATURES)
        & tasks["MaxNominalAttDistinctValues"].le(MAX_CATEGORY_LEVELS)
    )
    columns = [
        "tid",
        "did",
        "name",
        "target_feature",
        "NumberOfInstances",
        "NumberOfFeatures",
        "NumberOfClasses",
        "MinorityClassSize",
        "NumberOfMissingValues",
        "NumberOfNumericFeatures",
        "NumberOfSymbolicFeatures",
        "MaxNominalAttDistinctValues",
    ]
    candidates = tasks.loc[keep, columns].copy()
    candidates = candidates.sort_values(["name", "did", "tid"]).drop_duplicates("name")
    candidates = candidates.sample(frac=1, random_state=SEED).reset_index(drop=True)
    candidates.insert(0, "candidate_rank", np.arange(1, len(candidates) + 1))
    candidates.to_csv(path, index=False)
    return candidates


def _filled_numeric(X: pd.DataFrame, numeric: list[str]) -> pd.DataFrame:
    if not numeric:
        return pd.DataFrame(index=X.index)
    frame = X[numeric].astype(float).copy()
    for column in frame:
        median = frame[column].median()
        frame[column] = frame[column].fillna(0.0 if pd.isna(median) else median)
    return frame


def _fisher_ratios(frame: pd.DataFrame, y: np.ndarray) -> list[float]:
    ratios = []
    classes = np.unique(y)
    for column in frame:
        values = frame[column].to_numpy(dtype=float)
        grand_mean = values.mean()
        between = sum(
            (y == label).sum() * (values[y == label].mean() - grand_mean) ** 2
            for label in classes
        )
        within = sum(((values[y == label] - values[y == label].mean()) ** 2).sum() for label in classes)
        ratios.append(between / within if within > 0 else np.nan)
    return ratios


def manual_meta_features(
    X: pd.DataFrame, y: np.ndarray, categorical: list[str]
) -> dict[str, float]:
    n_rows, n_features = X.shape
    numeric = [column for column in X if column not in categorical]
    class_proportions = pd.Series(y).value_counts(normalize=True).sort_index().to_numpy()
    class_entropy = entropy(class_proportions)
    normalized_entropy = class_entropy / np.log(len(class_proportions))

    numeric_frame = _filled_numeric(X, numeric)
    skewness = numeric_frame.skew() if numeric else pd.Series(dtype=float)
    kurtosis = numeric_frame.kurt() if numeric else pd.Series(dtype=float)

    if len(numeric) > 1:
        correlation = numeric_frame.corr(method="spearman").to_numpy()
        upper = np.abs(correlation[np.triu_indices_from(correlation, k=1)])
        mean_abs_correlation = np.nanmean(upper) if np.isfinite(upper).any() else np.nan
    else:
        mean_abs_correlation = np.nan

    outlier_fractions = []
    for column in numeric_frame:
        values = numeric_frame[column]
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        outlier_fractions.append(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).mean())

    if numeric:
        scaled = StandardScaler().fit_transform(numeric_frame)
        pca = PCA(svd_solver="full").fit(scaled)
        variance = pca.explained_variance_ratio_
        first_component = variance[0]
        components_95 = (np.searchsorted(np.cumsum(variance), 0.95) + 1) / len(variance)
        fisher_ratios = _fisher_ratios(numeric_frame, y)
    else:
        first_component = components_95 = np.nan
        fisher_ratios = []

    category_cardinality = [X[column].nunique(dropna=True) / n_rows for column in categorical]
    unique_fractions = [X[column].nunique(dropna=False) / n_rows for column in X]

    return {
        "basic__n_instances": float(n_rows),
        "basic__n_features": float(n_features),
        "basic__n_numeric": float(len(numeric)),
        "basic__n_categorical": float(len(categorical)),
        "basic__n_classes": float(len(class_proportions)),
        "basic__missing_fraction": float(X.isna().to_numpy().mean()),
        "basic__feature_instance_ratio": n_features / n_rows,
        "stat__minority_fraction": float(class_proportions.min()),
        "stat__normalized_class_entropy": float(normalized_entropy),
        "stat__mean_abs_skewness": _safe_float(skewness.abs().mean()),
        "stat__mean_abs_kurtosis": _safe_float(kurtosis.abs().mean()),
        "stat__mean_abs_spearman": _safe_float(mean_abs_correlation),
        "stat__mean_outlier_fraction": _safe_float(np.mean(outlier_fractions)) if outlier_fractions else np.nan,
        "struct__duplicate_fraction": float(X.duplicated().mean()),
        "struct__mean_unique_fraction": _safe_float(np.mean(unique_fractions)),
        "struct__mean_category_cardinality": _safe_float(np.mean(category_cardinality)) if category_cardinality else np.nan,
        "struct__pca_first_component": _safe_float(first_component),
        "struct__pca_components_95_fraction": _safe_float(components_95),
        "struct__mean_fisher_ratio": _safe_float(np.nanmean(fisher_ratios)) if fisher_ratios else np.nan,
    }


def _pymfe_frame(X: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    clean = X.copy()
    for column in clean:
        if column in categorical:
            clean[column] = clean[column].astype(object).where(clean[column].notna(), "__missing__").astype(str)
        else:
            values = pd.to_numeric(clean[column], errors="coerce")
            clean[column] = values.fillna(0.0 if pd.isna(values.median()) else values.median())
    return clean


def _extract_pymfe_group(
    X: pd.DataFrame,
    y: np.ndarray,
    categorical: list[str],
    group: str,
) -> dict[str, float]:
    clean = _pymfe_frame(X, categorical)
    cat_indices = [clean.columns.get_loc(column) for column in categorical]
    extractor = MFE(
        groups=[group],
        features=PYMFE_FEATURES[group],
        summary=["mean", "sd"],
        suppress_warnings=True,
        random_state=SEED,
    )
    extractor.fit(
        clean.to_numpy(),
        y,
        cat_cols=cat_indices,
        transform_num=False,
        # Keeping every indicator avoids an arbitrary reference category.
        transform_cat="one-hot-full",
        rescale="standard",
        suppress_warnings=True,
    )
    names, values = extractor.extract(suppress_warnings=True)
    return {
        f"pymfe__{group}__{name.replace('.', '__')}": _safe_float(value)
        for name, value in zip(names, values)
    }


def _stratified_sample(
    X: pd.DataFrame, y: np.ndarray, max_rows: int
) -> tuple[pd.DataFrame, np.ndarray]:
    if len(X) <= max_rows:
        return X.reset_index(drop=True), np.asarray(y)
    selected, _ = train_test_split(
        np.arange(len(X)),
        train_size=max_rows,
        stratify=y,
        random_state=SEED,
    )
    return X.iloc[selected].reset_index(drop=True), np.asarray(y)[selected]


def pymfe_meta_features(
    X: pd.DataFrame,
    y: np.ndarray,
    categorical: list[str],
    complexity_max_rows: int = MAX_PYMFE_COMPLEXITY_ROWS,
) -> dict[str, float]:
    result = {}
    for group in ["general", "statistical"]:
        result.update(_extract_pymfe_group(X, y, categorical, group))

    sampled_X, sampled_y = _stratified_sample(X, y, complexity_max_rows)
    sampled_categorical = [column for column in categorical if column in sampled_X]
    result.update(_extract_pymfe_group(sampled_X, sampled_y, sampled_categorical, "complexity"))
    result["pymfe__complexity__sample_n"] = float(len(sampled_X))
    return result


def model_preprocessor(X: pd.DataFrame, categorical: list[str]) -> ColumnTransformer:
    numeric = [column for column in X if column not in categorical]
    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler(with_mean=False)),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=2,
                                sparse_output=True,
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers, sparse_threshold=1.0)


def benchmark_algorithms(
    X: pd.DataFrame, y: np.ndarray, categorical: list[str]
) -> tuple[dict[str, float], dict[str, list[float]], dict[str, float]]:
    model_X = X.copy()
    for column in categorical:
        model_X[column] = model_X[column].astype(object).where(model_X[column].notna(), np.nan)
        model_X[column] = model_X[column].map(lambda value: str(value) if pd.notna(value) else np.nan)

    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(model_X, y))
    means, fold_scores, durations = {}, {}, {}
    for name, estimator in ALGORITHMS.items():
        pipeline = Pipeline(
            [("prepare", model_preprocessor(model_X, categorical)), ("model", estimator)]
        )
        started = perf_counter()
        with threadpool_limits(limits=1):
            scores = cross_validate(
                pipeline,
                model_X,
                y,
                cv=folds,
                scoring="balanced_accuracy",
                n_jobs=1,
                error_score="raise",
            )["test_score"]
        durations[name] = perf_counter() - started
        fold_scores[name] = [float(value) for value in scores]
        means[name] = float(np.mean(scores))
    return means, fold_scores, durations


def _metric_key(prefix: str, name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.\-/]+", "_", name).strip("_")
    return f"{prefix}__{slug}"


def _completed_runs() -> pd.DataFrame:
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return pd.DataFrame()
    return mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            f"tags.run_type = 'dataset' AND "
            f"tags.pipeline_version = '{PIPELINE_VERSION}'"
        ),
        run_view_type=ViewType.ALL,
        max_results=100000,
        order_by=["start_time DESC"],
    )


def attempted_dataset_ids() -> tuple[set[int], set[int]]:
    runs = _completed_runs()
    if runs.empty:
        return set(), set()
    dataset_ids = pd.to_numeric(runs.get("params.dataset_id"), errors="coerce")
    status = runs.get("tags.status", pd.Series(index=runs.index, dtype=object))
    complete = set(dataset_ids[status.eq("complete")].dropna().astype(int))
    skipped = set(dataset_ids[status.eq("skipped")].dropna().astype(int))
    return complete, skipped


def run_dataset(candidate: pd.Series) -> bool:
    task_id = int(candidate["tid"])
    dataset_id = int(candidate["did"])
    run_name = f"did-{dataset_id}-{candidate['name']}"
    tags = {
        "run_type": "dataset",
        "status": "running",
        "pipeline_version": PIPELINE_VERSION,
        "candidate_rank": str(int(candidate["candidate_rank"])),
    }

    try:
        with mlflow.start_run(run_name=run_name, tags=tags):
            mlflow.log_params(
                {
                    "task_id": task_id,
                    "dataset_id": dataset_id,
                    "dataset_name": str(candidate["name"]),
                    "metric": "balanced_accuracy",
                    "cv_folds": N_FOLDS,
                    "algorithms": " | ".join(ALGORITHMS),
                    "pymfe_complexity_max_rows": MAX_PYMFE_COMPLEXITY_ROWS,
                }
            )
            started = perf_counter()
            try:
                X, y, categorical, numeric, info = load_openml_task(task_id)
            except DatasetSkip as error:
                mlflow.set_tags({"status": "skipped", "skip_reason": str(error)})
                return False
            load_seconds = perf_counter() - started

            feature_started = perf_counter()
            manual = manual_meta_features(X, y, categorical)
            pymfe = pymfe_meta_features(X, y, categorical)
            feature_seconds = perf_counter() - feature_started

            model_started = perf_counter()
            scores, fold_scores, durations = benchmark_algorithms(X, y, categorical)
            model_seconds = perf_counter() - model_started
            best_algorithm = max(ALGORITHMS, key=scores.get)

            metrics = {
                **{_metric_key("manual", key): value for key, value in manual.items()},
                **{key: value for key, value in pymfe.items()},
                **{_metric_key("score", key): value for key, value in scores.items()},
                **{_metric_key("fit_seconds", key): value for key, value in durations.items()},
                "load_seconds": load_seconds,
                "feature_seconds": feature_seconds,
                "model_seconds": model_seconds,
                "total_seconds": perf_counter() - started,
                "n_rows": float(len(X)),
                "n_features": float(X.shape[1]),
                "n_numeric": float(len(numeric)),
                "n_categorical": float(len(categorical)),
            }
            mlflow.log_metrics({key: value for key, value in metrics.items() if np.isfinite(value)})
            mlflow.log_params(
                {
                    "target_name": info["target_name"],
                    "best_algorithm": best_algorithm,
                    "n_dropped_constant": info["n_dropped_constant"],
                }
            )
            mlflow.set_tags({"status": "complete", "best_algorithm": best_algorithm})
            record = {
                **info,
                "categorical_columns": categorical,
                "numeric_columns": numeric,
                "manual_meta_features": manual,
                "pymfe_meta_features": pymfe,
                "algorithm_scores": scores,
                "fold_scores": fold_scores,
                "fit_seconds": durations,
            }
            mlflow.log_dict(record, "dataset_result.json")
        return True
    except Exception:
        # The failed run is preserved by MLflow and will be retried on the next invocation.
        print(traceback.format_exc(), flush=True)
        return False


def export_completed_runs() -> pd.DataFrame:
    configure_tracking()
    runs = _completed_runs()
    if runs.empty:
        return pd.DataFrame()
    runs = runs[runs.get("tags.status").eq("complete")].copy()
    if runs.empty:
        return pd.DataFrame()
    runs["dataset_id"] = pd.to_numeric(runs["params.dataset_id"], errors="coerce")
    runs = runs.sort_values("start_time", ascending=False).drop_duplicates("dataset_id")

    result = pd.DataFrame(
        {
            "run_id": runs["run_id"],
            "task_id": pd.to_numeric(runs["params.task_id"], errors="coerce").astype("Int64"),
            "dataset_id": runs["dataset_id"].astype("Int64"),
            "dataset_name": runs["params.dataset_name"],
            "target_name": runs.get("params.target_name"),
            "best_algorithm": runs["params.best_algorithm"],
            "candidate_rank": pd.to_numeric(runs.get("tags.candidate_rank"), errors="coerce"),
        }
    )
    for column in runs.columns:
        if column.startswith("metrics."):
            result[column.removeprefix("metrics.")] = pd.to_numeric(runs[column], errors="coerce")
    result = result.sort_values(["candidate_rank", "dataset_id"]).reset_index(drop=True)
    result.to_csv(RESULTS_DIR / "meta_dataset_from_mlflow.csv", index=False)
    return result


def _category_relabel(
    X: pd.DataFrame, y: np.ndarray, categorical: list[str]
) -> tuple[pd.DataFrame, np.ndarray]:
    changed = X.copy()
    rng = np.random.default_rng(SEED + 91)
    for column in categorical:
        levels = pd.unique(changed[column].dropna())
        replacements = [f"category_{i}" for i in rng.permutation(len(levels))]
        changed[column] = changed[column].map(dict(zip(levels, replacements)))
    target_levels = np.unique(y)
    target_map = dict(zip(target_levels, rng.permutation(target_levels)))
    return changed, np.array([target_map[value] for value in y])


def invariance_comparison(
    X: pd.DataFrame, y: np.ndarray, categorical: list[str]
) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 17)
    relabeled_X, relabeled_y = _category_relabel(X, y, categorical)
    variants = {
        "row order": (X.iloc[rng.permutation(len(X))].reset_index(drop=True), y[rng.permutation(len(y))]),
        "column order": (X[X.columns[::-1]].copy(), y.copy()),
        "category labels": (relabeled_X, relabeled_y),
    }
    # Use one shared row permutation for X and y.
    row_order = rng.permutation(len(X))
    variants["row order"] = (X.iloc[row_order].reset_index(drop=True), y[row_order])

    extractors = {
        "Manual": lambda frame, target, cats: manual_meta_features(frame, target, cats),
        "PyMFE": lambda frame, target, cats: pymfe_meta_features(
            frame, target, cats, complexity_max_rows=len(frame)
        ),
    }
    records = []
    for extractor_name, extractor in extractors.items():
        baseline = extractor(X, y, categorical)
        for variant_name, (changed_X, changed_y) in variants.items():
            changed_categorical = [column for column in changed_X if column in categorical]
            values = extractor(changed_X, changed_y, changed_categorical)
            for feature in sorted(set(baseline) | set(values)):
                original = baseline.get(feature, np.nan)
                transformed = values.get(feature, np.nan)
                close = bool(np.isclose(original, transformed, rtol=1e-7, atol=1e-9, equal_nan=True))
                records.append(
                    {
                        "Extractor": extractor_name,
                        "Variant": variant_name,
                        "Feature": feature,
                        "Original": original,
                        "Transformed": transformed,
                        "Absolute difference": abs(original - transformed)
                        if np.isfinite(original) and np.isfinite(transformed)
                        else np.nan,
                        "Invariant": close,
                    }
                )
    return pd.DataFrame(records)


def run_invariance_check(force: bool = False) -> pd.DataFrame:
    configure_tracking()
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    previous = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            "tags.run_type = 'invariance' AND tags.status = 'complete' AND "
            f"tags.pipeline_version = '{PIPELINE_VERSION}'"
        ),
        max_results=10,
    )
    path = RESULTS_DIR / "invariance_check.csv"
    if not force and not previous.empty and path.exists():
        return pd.read_csv(path)

    X, y, categorical, _, info = load_openml_task(31)
    comparison = invariance_comparison(X, y, categorical)
    comparison.to_csv(path, index=False)
    summary = comparison.groupby(["Extractor", "Variant"], as_index=False).agg(
        features=("Feature", "size"),
        invariant=("Invariant", "sum"),
        max_abs_difference=("Absolute difference", "max"),
    )
    with mlflow.start_run(
        run_name="invariance-credit-g",
        tags={
            "run_type": "invariance",
            "status": "complete",
            "pipeline_version": PIPELINE_VERSION,
        },
    ):
        mlflow.log_params(
            {"task_id": 31, "dataset_id": info["dataset_id"], "dataset_name": info["dataset_name"]}
        )
        for row in summary.itertuples(index=False):
            stem = f"{row.Extractor}_{row.Variant}".lower().replace(" ", "_")
            mlflow.log_metric(f"{stem}__invariant_fraction", row.invariant / row.features)
            if np.isfinite(row.max_abs_difference):
                mlflow.log_metric(f"{stem}__max_abs_difference", row.max_abs_difference)
        mlflow.log_artifact(str(path))
    return comparison


def run_experiments(
    target: int = TARGET_DATASETS,
    max_new: int | None = None,
    refresh_catalog: bool = False,
) -> pd.DataFrame:
    configure_tracking()
    candidates = candidate_tasks(refresh=refresh_catalog)
    complete, skipped = attempted_dataset_ids()
    new_attempts = 0
    print(f"Found {len(complete)} completed datasets and {len(skipped)} deterministic skips.", flush=True)

    for candidate in candidates.itertuples(index=False):
        if len(complete) >= target or (max_new is not None and new_attempts >= max_new):
            break
        row = pd.Series(candidate._asdict())
        dataset_id = int(row["did"])
        if dataset_id in complete or dataset_id in skipped:
            continue
        new_attempts += 1
        print(
            f"[{len(complete) + 1}/{target}] {row['name']} (task {int(row['tid'])}, dataset {dataset_id})",
            flush=True,
        )
        if run_dataset(row):
            complete.add(dataset_id)
        else:
            skipped.add(dataset_id)

    result = export_completed_runs()
    print(f"Exported {len(result)} completed dataset runs to {RESULTS_DIR}.", flush=True)
    return result


def load_invariance_summary() -> pd.DataFrame:
    path = RESULTS_DIR / "invariance_check.csv"
    if not path.exists():
        raise FileNotFoundError("Run the invariance check before building the report.")
    detail = pd.read_csv(path)
    return detail.groupby(["Extractor", "Variant"], as_index=False).agg(
        Features=("Feature", "size"),
        Invariant=("Invariant", "sum"),
        max_abs_difference=("Absolute difference", "max"),
    )


def meta_dataset_features(
    meta_dataset: pd.DataFrame, feature_columns: list[str], target_column: str = "best_algorithm"
) -> tuple[dict[str, float], dict[str, float]]:
    X = meta_dataset[feature_columns].replace([np.inf, -np.inf], np.nan)
    X = X.apply(lambda column: column.fillna(column.median()))
    y, _ = pd.factorize(meta_dataset[target_column], sort=True)
    manual = manual_meta_features(X, y, categorical=[])
    pymfe = pymfe_meta_features(X, y, categorical=[], complexity_max_rows=len(X))
    return manual, pymfe


def verify_score_selection(meta_dataset: pd.DataFrame) -> bool:
    score_columns = [_metric_key("score", name) for name in ALGORITHMS]
    selected = meta_dataset[score_columns].idxmax(axis=1).str.removeprefix("score__").str.replace("_", " ")
    normalized = meta_dataset["best_algorithm"].str.lower()
    return bool(selected.str.lower().eq(normalized).all())
