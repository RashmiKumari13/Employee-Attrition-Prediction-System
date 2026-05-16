import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

from config import METADATA_PATH, MODEL_PATH


class AttritionModelService:
    def __init__(self, model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model_bundle: dict[str, Any] | None = None
        self.metadata: dict[str, Any] = {}
        self.pipeline = None
        self.explainer = None
        self.reload()

    @property
    def is_ready(self) -> bool:
        return self.model_bundle is not None and self.pipeline is not None and self.explainer is not None

    def reload(self) -> None:
        self.model_bundle = None
        self.metadata = {}
        self.pipeline = None
        self.explainer = None

        if not self.model_path.exists() or not self.metadata_path.exists():
            return

        self.model_bundle = joblib.load(self.model_path)
        self.pipeline = self.model_bundle["pipeline"]
        model = self.pipeline.named_steps["model"]
        self.explainer = shap.TreeExplainer(model)

        with self.metadata_path.open("r", encoding="utf-8") as metadata_file:
            self.metadata = json.load(metadata_file)

    def get_feature_schema(self) -> list[dict[str, Any]]:
        return self.metadata.get("feature_schema", [])

    def _validate_ready(self) -> None:
        if not self.is_ready:
            raise RuntimeError(
                "Model artifacts are missing. Run `python backend/train_model.py` to create model files."
            )

    def _coerce_value(self, raw_value: Any, feature_spec: dict[str, Any]) -> Any:
        if raw_value is None or raw_value == "":
            raw_value = feature_spec.get("default")

        if feature_spec["kind"] == "numeric":
            numeric_value = float(raw_value)
            if feature_spec.get("dtype") == "integer":
                return int(round(numeric_value))
            return numeric_value

        return str(raw_value)

    def _build_input_row(self, payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        row: dict[str, Any] = {}
        warnings: list[str] = []
        feature_schema = self.get_feature_schema()

        for feature_spec in feature_schema:
            feature_name = feature_spec["name"]
            raw_value = payload.get(feature_name, feature_spec.get("default"))
            value = self._coerce_value(raw_value, feature_spec)

            if feature_spec["kind"] == "categorical":
                options = feature_spec.get("options", [])
                if options and value not in options:
                    warnings.append(
                        f"{feature_name}: '{value}' is unseen during training. Prediction still computed."
                    )

            row[feature_name] = value

        return row, warnings

    @staticmethod
    def _extract_expected_value(expected_value: Any) -> float:
        if isinstance(expected_value, (list, tuple, np.ndarray)):
            flattened = np.asarray(expected_value).astype(float).flatten()
            if flattened.size == 0:
                return 0.0
            return float(flattened[1] if flattened.size > 1 else flattened[0])
        return float(expected_value)

    @staticmethod
    def _extract_shap_row(shap_values: Any) -> np.ndarray:
        if isinstance(shap_values, list):
            if len(shap_values) > 1:
                row = np.asarray(shap_values[1])[0]
            else:
                row = np.asarray(shap_values[0])[0]
            return np.asarray(row, dtype=float)

        values = np.asarray(shap_values)
        if values.ndim == 3:
            index = 1 if values.shape[2] > 1 else 0
            return values[0, :, index].astype(float)
        if values.ndim == 2:
            return values[0].astype(float)
        return values.astype(float).flatten()

    def _build_explainability_report(self, sample_frame: pd.DataFrame) -> dict[str, Any]:
        transformed = self.pipeline.named_steps["preprocess"].transform(sample_frame)
        shap_values = self.explainer.shap_values(transformed)
        row_values = self._extract_shap_row(shap_values)
        base_value = self._extract_expected_value(self.explainer.expected_value)

        feature_names = self.model_bundle["encoded_feature_names"]
        feature_map = self.model_bundle["encoded_feature_map"]
        aggregated: dict[str, float] = {}

        for index, shap_value in enumerate(row_values):
            feature_name = feature_names[index]
            original_feature = feature_map.get(feature_name, feature_name)
            aggregated[original_feature] = aggregated.get(original_feature, 0.0) + float(shap_value)

        sorted_contributors = sorted(aggregated.items(), key=lambda item: abs(item[1]), reverse=True)
        top_contributors = [
            {
                "feature": feature,
                "shap_value": round(value, 5),
                "direction": "increase_risk" if value >= 0 else "decrease_risk",
            }
            for feature, value in sorted_contributors[:8]
        ]

        return {
            "base_value": round(base_value, 5),
            "raw_output_estimate": round(base_value + float(row_values.sum()), 5),
            "top_contributors": top_contributors,
        }

    @staticmethod
    def _readable_feature_name(feature_name: str) -> str:
        return re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", feature_name).replace("_", " ").strip()

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _build_recommendations(
        self,
        input_row: dict[str, Any],
        prediction: int,
        risk_probability: float,
        explainability: dict[str, Any],
    ) -> tuple[str, list[str]]:
        top_contributors = explainability.get("top_contributors", [])
        positive_factors = [
            contributor["feature"]
            for contributor in top_contributors
            if contributor.get("direction") == "increase_risk"
        ]

        if prediction == 1 and positive_factors:
            top_factor_labels = [self._readable_feature_name(name) for name in positive_factors[:2]]
            summary = f"Employee likely to leave due to {' + '.join(top_factor_labels)}."
        elif prediction == 1:
            summary = "Employee likely to leave; risk elevated based on current profile."
        elif positive_factors:
            summary = (
                f"Employee currently stable, but monitor {self._readable_feature_name(positive_factors[0])}."
            )
        else:
            summary = "Employee currently appears stable with low near-term attrition risk."

        recommendations: list[str] = []

        if str(input_row.get("OverTime", "No")) == "Yes":
            recommendations.append(
                "Reduce recurring overtime through workload balancing and manager check-ins."
            )

        if self._safe_float(input_row.get("JobSatisfaction"), 3) <= 2:
            recommendations.append(
                "Create a role-specific engagement plan to improve job satisfaction."
            )

        if self._safe_float(input_row.get("WorkLifeBalance"), 3) <= 2:
            recommendations.append(
                "Introduce flexible scheduling and wellness support to improve work-life balance."
            )

        if self._safe_float(input_row.get("EnvironmentSatisfaction"), 3) <= 2:
            recommendations.append(
                "Address workplace environment pain points with targeted team interventions."
            )

        if self._safe_float(input_row.get("TrainingTimesLastYear"), 3) <= 1:
            recommendations.append(
                "Increase training frequency and personalized upskilling opportunities."
            )

        if self._safe_float(input_row.get("YearsSinceLastPromotion"), 1) >= 4:
            recommendations.append(
                "Review promotion readiness and define a clear advancement path."
            )

        median_income = self.model_bundle["feature_defaults"].get("MonthlyIncome", 5000)
        if self._safe_float(input_row.get("MonthlyIncome"), median_income) < float(median_income):
            recommendations.append(
                "Benchmark compensation and consider salary or incentive adjustments."
            )

        if not recommendations and prediction == 1:
            recommendations.append(
                "Set a 30-day retention action plan with HR and the direct manager."
            )

        if not recommendations and risk_probability < 0.2:
            recommendations.append(
                "Maintain current engagement practices and monitor risk monthly."
            )

        if not recommendations:
            recommendations.append(
                "Continue proactive one-on-ones and monitor key engagement indicators."
            )

        return summary, recommendations

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._validate_ready()

        input_row, warnings = self._build_input_row(payload)
        feature_order = self.model_bundle["feature_columns"]
        sample_frame = pd.DataFrame([input_row], columns=feature_order)

        probabilities = self.pipeline.predict_proba(sample_frame)[0]
        class_labels = list(self.pipeline.named_steps["model"].classes_)
        positive_index = class_labels.index(1)
        risk_probability = float(probabilities[positive_index])
        prediction = int(self.pipeline.predict(sample_frame)[0])
        explainability = self._build_explainability_report(sample_frame)
        summary, recommendations = self._build_recommendations(
            input_row=input_row,
            prediction=prediction,
            risk_probability=risk_probability,
            explainability=explainability,
        )

        return {
            "prediction": prediction,
            "risk_label": "High Risk" if prediction == 1 else "Low Risk",
            "attrition_probability": round(risk_probability, 4),
            "attrition_percent": round(risk_probability * 100, 2),
            "warnings": warnings,
            "input_features": input_row,
            "explainability": explainability,
            "summary": summary,
            "recommendations": recommendations,
        }
