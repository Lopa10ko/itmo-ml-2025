from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lab2_experiment import SEED, meta_dataset_features

META_MODELS = {
    "Most frequent": DummyClassifier(strategy="most_frequent"),
    "Logistic regression": LogisticRegression(
        class_weight="balanced", max_iter=3000, random_state=SEED
    ),
    "15-nearest neighbors": KNeighborsClassifier(n_neighbors=15, weights="distance"),
    "Random forest": RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=1,
    ),
}
MIN_OBSERVED_FRACTION = 0.20
PYMFE_NUMERICAL_ARTIFACTS = {
    "pymfe__statistical__mean__mean",
    "pymfe__statistical__mean__sd",
    "pymfe__statistical__sd__mean",
    "pymfe__statistical__sd__sd",
}


def feature_columns(meta_dataset: pd.DataFrame) -> tuple[list[str], list[str]]:
    manual_candidates = sorted(column for column in meta_dataset if column.startswith("manual__"))
    pymfe_candidates = sorted(
        column
        for column in meta_dataset
        if column.startswith("pymfe__")
        and not column.endswith("sample_n")
        and column not in PYMFE_NUMERICAL_ARTIFACTS
    )
    manual = [
        column
        for column in manual_candidates
        if meta_dataset[column].notna().mean() >= MIN_OBSERVED_FRACTION
    ]
    pymfe = [
        column
        for column in pymfe_candidates
        if meta_dataset[column].notna().mean() >= MIN_OBSERVED_FRACTION
    ]
    return manual, pymfe


def extreme_feature_rows(
    meta_dataset: pd.DataFrame, columns: list[str], top_n: int = 8
) -> pd.DataFrame:
    values = meta_dataset[columns].replace([np.inf, -np.inf], np.nan)
    imputed = values.fillna(values.median())
    medians = imputed.median()
    mad = (imputed - medians).abs().median()
    usable = mad > np.finfo(float).eps
    robust_z = 0.67448975 * (imputed.loc[:, usable] - medians[usable]).div(mad[usable])
    magnitude = robust_z.abs()
    selected = magnitude.max(axis=1).nlargest(top_n).index
    driving_feature = magnitude.loc[selected].idxmax(axis=1)

    result = meta_dataset.loc[
        selected, ["dataset_id", "dataset_name", "best_algorithm"]
    ].copy()
    result["robust_z"] = [robust_z.at[index, driving_feature.at[index]] for index in selected]
    result["driving_feature"] = driving_feature.str.removeprefix("pymfe__")
    result["feature_value"] = [imputed.at[index, driving_feature.at[index]] for index in selected]
    return result.reset_index(drop=True)


def _matrix(meta_dataset: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, list[str]]:
    values = meta_dataset[columns].replace([np.inf, -np.inf], np.nan)
    missing = values.isna()
    indicator_columns = [column for column in columns if missing[column].any()]
    imputed = SimpleImputer(strategy="median").fit_transform(values)
    if indicator_columns:
        indicators = missing[indicator_columns].astype(float).to_numpy()
        imputed = np.column_stack([imputed, indicators])
    names = columns + [f"missing__{column}" for column in indicator_columns]
    return StandardScaler().fit_transform(imputed), names


def pca_projection(meta_dataset: pd.DataFrame, columns: list[str]):
    matrix, matrix_columns = _matrix(meta_dataset, columns)
    pca = PCA(n_components=2, random_state=SEED)
    coordinates = pca.fit_transform(matrix)
    projection = meta_dataset[["dataset_id", "dataset_name", "best_algorithm"]].copy()
    projection["PC1"] = coordinates[:, 0]
    projection["PC2"] = coordinates[:, 1]
    loadings = pd.DataFrame(
        pca.components_.T,
        index=matrix_columns,
        columns=["PC1", "PC2"],
    ).rename_axis("feature").reset_index()
    return projection, pca.explained_variance_ratio_, loadings


def evaluate_meta_models(
    meta_dataset: pd.DataFrame, columns: list[str], source: str
) -> pd.DataFrame:
    X = meta_dataset[columns].replace([np.inf, -np.inf], np.nan)
    y = meta_dataset["best_algorithm"]
    min_class = int(y.value_counts().min())
    if min_class < 2:
        raise ValueError("Every meta-class needs at least two datasets for validation.")
    n_splits = min(5, min_class)
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=5,
        random_state=SEED,
    )
    scoring = {
        "balanced_accuracy": "balanced_accuracy",
        "accuracy": "accuracy",
        "macro_f1": make_scorer(f1_score, average="macro", zero_division=0),
    }

    records = []
    for model_name, model in META_MODELS.items():
        pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                ("model", model),
            ]
        )
        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )
        record = {
            "Feature source": source,
            "Meta-model": model_name,
            "Splits": len(scores["test_balanced_accuracy"]),
        }
        for metric in scoring:
            values = scores[f"test_{metric}"]
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_sd"] = float(values.std(ddof=1))
        records.append(record)
    return pd.DataFrame(records)


def evaluate_both_sources(meta_dataset: pd.DataFrame) -> pd.DataFrame:
    manual, pymfe = feature_columns(meta_dataset)
    result = pd.concat(
        [
            evaluate_meta_models(meta_dataset, manual, "Manual"),
            evaluate_meta_models(meta_dataset, pymfe, "PyMFE"),
        ],
        ignore_index=True,
    )
    return result


def characterize_meta_datasets(meta_dataset: pd.DataFrame) -> pd.DataFrame:
    manual_columns, pymfe_columns = feature_columns(meta_dataset)
    manual_manual, manual_pymfe = meta_dataset_features(meta_dataset, manual_columns)
    pymfe_manual, pymfe_pymfe = meta_dataset_features(meta_dataset, pymfe_columns)

    rows = []
    for source, extracted in [
        ("Manual meta-dataset / manual extractor", manual_manual),
        ("Manual meta-dataset / PyMFE extractor", manual_pymfe),
        ("PyMFE meta-dataset / manual extractor", pymfe_manual),
        ("PyMFE meta-dataset / PyMFE extractor", pymfe_pymfe),
    ]:
        row = {"Object": source}
        row.update(extracted)
        rows.append(row)
    return pd.DataFrame(rows)
