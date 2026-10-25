from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
MODEL_PATH = ROOT / "models" / "readmission_model.joblib"

app = Flask(__name__, static_folder=str(FRONTEND))

bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
threshold = float(bundle["threshold"])
features = bundle["feature_columns"]


@app.get("/")
def index():
    return send_from_directory(FRONTEND, "index.html")


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model": bundle["model_name"],
        "features": features,
        "threshold": threshold
    })


@app.post("/api/predict")
def predict():
    body = request.get_json(silent=True)

    if not isinstance(body, dict):
        return jsonify({"error": "Send patient details as JSON."}), 400

    missing = [name for name in features if name not in body]
    if missing:
        return jsonify({"error": "Missing input fields.", "missing": missing}), 400

    try:
        row = {}
        for name in features:
            value = body[name]
            if value is None or str(value).strip() == "":
                return jsonify({"error": f"{name} is required."}), 400
            if name in {
                "age", "time_in_hospital", "previous_admissions",
                "num_lab_procedures", "num_medications"
            }:
                row[name] = float(value)
            else:
                row[name] = str(value).strip()

        patient = pd.DataFrame([row], columns=features)
        probability = float(pipeline.predict_proba(patient)[0][1])
        prediction = int(probability >= threshold)

        return jsonify({
            "success": True,
            "model": bundle["model_name"],
            "risk_probability": round(probability, 4),
            "risk_percent": round(probability * 100, 2),
            "threshold": threshold,
            "prediction": prediction,
            "result": (
                "Higher readmission risk"
                if prediction else "Lower readmission risk"
            ),
            "note": "Research/demo estimate only; not a clinical diagnosis."
        })
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid patient input: {exc}"}), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed. Check server logs."}), 500


@app.get("/api/metrics")
def metrics():
    path = ROOT / "models" / "final_metrics.json"
    if not path.exists():
        return jsonify({"error": "Metrics file not found."}), 404
    return app.response_class(path.read_text(), mimetype="application/json")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
