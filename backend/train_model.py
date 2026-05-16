import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import DATA_PATH, METADATA_PATH, MODEL_PATH

TARGET_COLUMN = "Attrition"
DROP_COLUMNS = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]


def build_feature_schema(frame: pd.DataFrame) -> list[dict]:
    schema = []

    for column in frame.columns:
        column_series = frame[column]

        if pd.api.types.is_numeric_dtype(column_series):
            values = column_series.dropna()
            dtype = "integer" if pd.api.types.is_integer_dtype(column_series) else "float"
            median = float(values.median()) if not values.empty else 0.0
            default_value = int(round(median)) if dtype == "integer" else round(median, 3)

            schema.append(
                {
                    "name": column,
                    "kind": "numeric",
                    "dtype": dtype,
                    "default": default_value,
                    "min": float(values.min()) if not values.empty else 0.0,
                    "max": float(values.max()) if not values.empty else 0.0,
                }
            )
        else:
            clean_values = column_series.dropna().astype(str)
            options = sorted(clean_values.unique().tolist())
            mode = clean_values.mode()
            default_value = mode.iloc[0] if not mode.empty else ""

            schema.append(
                {
                    "name": column,
                    "kind": "categorical",
                    "default": default_value,
                    "options": options,
                }
            )

    return schema


def simplify_feature_name(encoded_name: str, categorical_features: list[str]) -> str:
    if encoded_name.startswith("num__"):
        return encoded_name.replace("num__", "", 1)

    if encoded_name.startswith("cat__"):
        compact_name = encoded_name.replace("cat__", "", 1)
        for feature_name in categorical_features:
            if compact_name == feature_name or compact_name.startswith(f"{feature_name}_"):
                return feature_name
        return compact_name

    return encoded_name


def train_and_save(data_path: Path, model_path: Path, metadata_path: Path) -> dict:
    dataset = pd.read_csv(data_path)
    dropped_columns = [col for col in DROP_COLUMNS if col in dataset.columns]
    dataset = dataset.drop(columns=dropped_columns)

    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    X = dataset.drop(columns=[TARGET_COLUMN]).copy()
    y = dataset[TARGET_COLUMN].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError("Target column contains unknown values. Expected only 'Yes' and 'No'.")

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=2,
        subsample=0.85,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    test_accuracy = accuracy_score(y_test, predictions)
    test_auc = roc_auc_score(y_test, probabilities)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")

    report = classification_report(y_test, predictions, output_dict=True)

    fitted_preprocessor = pipeline.named_steps["preprocess"]
    encoded_feature_names = fitted_preprocessor.get_feature_names_out().tolist()
    encoded_feature_map = {
        encoded_name: simplify_feature_name(encoded_name, categorical_features)
        for encoded_name in encoded_feature_names
    }

    feature_schema = build_feature_schema(X)
    feature_defaults = {entry["name"]: entry["default"] for entry in feature_schema}

    model_bundle = {
        "pipeline": pipeline,
        "feature_columns": X.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "encoded_feature_names": encoded_feature_names,
        "encoded_feature_map": encoded_feature_map,
        "feature_defaults": feature_defaults,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, model_path)

    metrics = {
        "test_accuracy": round(float(test_accuracy), 4),
        "test_roc_auc": round(float(test_auc), 4),
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
        "benchmark_target_accuracy": 0.89,
        "classification_report": report,
    }

    metadata = {
        "project_name": "Employee Attrition Prediction System",
        "dataset": "IBM HR Analytics Employee Attrition & Performance",
        "algorithm": "GradientBoostingClassifier",
        "dropped_columns": dropped_columns,
        "rows_total": int(len(dataset)),
        "feature_count": int(X.shape[1]),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "feature_schema": feature_schema,
        "metrics": metrics,
    }

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train gradient boosting model for attrition prediction.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help="Path to IBM attrition CSV file.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help="Output path for serialized model bundle.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=METADATA_PATH,
        help="Output path for metadata and metrics JSON.",
    )
    args = parser.parse_args()

    metadata = train_and_save(args.data_path, args.model_path, args.metadata_path)
    metrics = metadata["metrics"]

    print("Model training completed successfully.")
    print(f"Test Accuracy: {metrics['test_accuracy']}")
    print(f"Cross-Validation Accuracy (mean): {metrics['cv_accuracy_mean']}")
    print(f"Test ROC AUC: {metrics['test_roc_auc']}")
    print(f"Model saved to: {args.model_path}")
    print(f"Metadata saved to: {args.metadata_path}")


if __name__ == "__main__":
    main()
