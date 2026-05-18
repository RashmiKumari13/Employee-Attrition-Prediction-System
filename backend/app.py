from pathlib import Path
from datetime import datetime
from io import BytesIO

from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS

from model_service import AttritionModelService
from report_service import build_explainability_pdf

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

model_service = AttritionModelService()
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
FRONTEND_BASE_PREFIX = "Employee-Attrition-Prediction-System"


@app.get("/")
def root():
    if (FRONTEND_DIST / "index.html").exists():
        return send_from_directory(FRONTEND_DIST, "index.html")

    return jsonify(
        {
            "message": "Employee Attrition Prediction API is running.",
            "tip": "Build frontend (`npm run build`) to serve dashboard from :5000 root.",
            "api_endpoints": [
                "/api/health",
                "/api/model-info",
                "/api/features",
                "/api/predict",
                "/api/report",
                "/api/reload-model",
            ],
        }
    )


@app.get(f"/{FRONTEND_BASE_PREFIX}/")
def root_with_base_prefix():
    if (FRONTEND_DIST / "index.html").exists():
        return send_from_directory(FRONTEND_DIST, "index.html")

    return jsonify(
        {
            "message": "Employee Attrition Prediction API is running.",
            "tip": "Build frontend (`npm run build`) to serve dashboard from :5000 root.",
            "api_endpoints": [
                "/api/health",
                "/api/model-info",
                "/api/features",
                "/api/predict",
                "/api/report",
                "/api/reload-model",
            ],
        }
    )


@app.get(f"/{FRONTEND_BASE_PREFIX}/<path:path>")
def serve_prefixed_frontend_asset(path: str):
    candidate = FRONTEND_DIST / path
    if candidate.exists() and candidate.is_file():
        return send_from_directory(FRONTEND_DIST, path)

    return jsonify({"error": "Asset not found."}), 404


@app.get("/api/health")
def health_check():
    return jsonify(
        {
            "status": "ok",
            "model_ready": model_service.is_ready,
            "project": "Employee Attrition Prediction System",
        }
    )


@app.get("/api/model-info")
def model_info():
    if not model_service.is_ready:
        return (
            jsonify(
                {
                    "model_ready": False,
                    "message": "Model not trained yet. Run `python backend/train_model.py` first.",
                }
            ),
            503,
        )

    metadata = model_service.metadata
    response = {
        "model_ready": True,
        "project_name": metadata.get("project_name"),
        "dataset": metadata.get("dataset"),
        "algorithm": metadata.get("algorithm"),
        "rows_total": metadata.get("rows_total"),
        "feature_count": metadata.get("feature_count"),
        "metrics": metadata.get("metrics"),
    }
    return jsonify(response)


@app.get("/api/features")
def feature_schema():
    if not model_service.is_ready:
        return (
            jsonify(
                {
                    "model_ready": False,
                    "message": "Model not trained yet. Run `python backend/train_model.py` first.",
                    "features": [],
                }
            ),
            503,
        )

    return jsonify({"model_ready": True, "features": model_service.get_feature_schema()})


@app.post("/api/predict")
def predict():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON."}), 400

        payload = request.get_json() or {}
        prediction = model_service.predict(payload)
        return jsonify(prediction)
    except RuntimeError as runtime_error:
        return jsonify({"error": str(runtime_error)}), 503
    except ValueError as value_error:
        return jsonify({"error": f"Invalid input: {value_error}"}), 400
    except Exception as unexpected_error:  # pragma: no cover
        return jsonify({"error": f"Unexpected server error: {unexpected_error}"}), 500


@app.post("/api/reload-model")
def reload_model():
    model_service.reload()
    return jsonify({"model_ready": model_service.is_ready})


@app.post("/api/report")
def download_pdf_report():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON."}), 400

        payload = request.get_json() or {}
        employee_name = payload.get("employee_name")
        prediction = model_service.predict(payload)
        pdf_bytes = build_explainability_pdf(prediction=prediction, employee_name=employee_name)
        filename_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        return send_file(
            BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"attrition_explainability_report_{filename_stamp}.pdf",
        )
    except RuntimeError as runtime_error:
        return jsonify({"error": str(runtime_error)}), 503
    except ValueError as value_error:
        return jsonify({"error": f"Invalid input: {value_error}"}), 400
    except Exception as unexpected_error:  # pragma: no cover
        return jsonify({"error": f"Unexpected server error: {unexpected_error}"}), 500


@app.get("/assets/<path:filename>")
def serve_frontend_assets(filename: str):
    assets_path = FRONTEND_DIST / "assets"
    return send_from_directory(assets_path, filename)


@app.get("/<path:path>")
def spa_fallback(path: str):
    if path.startswith("api/"):
        return jsonify({"error": "API route not found."}), 404

    candidate = FRONTEND_DIST / path
    if candidate.exists() and candidate.is_file():
        return send_from_directory(FRONTEND_DIST, path)

    if (FRONTEND_DIST / "index.html").exists():
        return send_from_directory(FRONTEND_DIST, "index.html")

    return jsonify({"error": f"Path not found: /{path}"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
