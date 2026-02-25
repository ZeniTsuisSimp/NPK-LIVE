"""
Model Evaluation Module
- Loads trained model and test data
- Computes classification metrics
- Saves metrics to reports/metrics.json
- Logs metrics to MLflow
"""

import os
import json
import yaml
import numpy as np
import joblib
import mlflow
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


def load_params(params_path="params.yaml"):
    """Load parameters from params.yaml."""
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def evaluate_model(model_path, data_dir="data/processed", reports_dir="reports"):
    """Evaluate the trained model and generate metrics."""
    # Load test data
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    metadata = joblib.load(os.path.join(data_dir, "metadata.pkl"))

    # Load model bundle
    bundle = joblib.load(model_path)
    model = bundle["model"]

    # Predictions
    y_pred = model.predict(X_test)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Classification report
    report = classification_report(
        y_test, y_pred,
        target_names=metadata["target_names"],
        output_dict=True,
    )

    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "classification_report": report,
    }

    # Save metrics
    os.makedirs(reports_dir, exist_ok=True)
    metrics_path = os.path.join(reports_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"{'='*50}")
    print(f"  Model Evaluation Results")
    print(f"{'='*50}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"{'='*50}")
    print(f"\nDetailed report saved to {metrics_path}")

    # Print classification report
    print(f"\n{classification_report(y_test, y_pred, target_names=metadata['target_names'])}")

    return metrics


def main():
    """Run evaluation with MLflow logging."""
    params = load_params()
    mlflow_config = params["mlflow"]

    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
    mlflow.set_experiment(mlflow_config["experiment_name"])

    metrics = evaluate_model("models/npk_crop_model.pkl")

    # Log to MLflow
    with mlflow.start_run(run_name="rf-evaluation"):
        mlflow.log_metrics({
            "eval_accuracy": metrics["accuracy"],
            "eval_precision": metrics["precision"],
            "eval_recall": metrics["recall"],
            "eval_f1_score": metrics["f1_score"],
        })
        mlflow.log_artifact("reports/metrics.json")
        print(f"\n✅ Evaluation logged to MLflow.")


if __name__ == "__main__":
    main()
