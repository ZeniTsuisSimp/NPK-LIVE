"""
Model Training Module
- Loads preprocessed data
- Trains RandomForestClassifier with params from params.yaml
- Logs experiment to MLflow
- Saves model bundle (.pkl)
"""

import os
import yaml
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def load_params(params_path="params.yaml"):
    """Load parameters from params.yaml."""
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def load_preprocessed(input_dir="data/processed"):
    """Load preprocessed data artifacts."""
    data = {
        "X_train": np.load(os.path.join(input_dir, "X_train.npy")),
        "X_test": np.load(os.path.join(input_dir, "X_test.npy")),
        "y_train": np.load(os.path.join(input_dir, "y_train.npy")),
        "y_test": np.load(os.path.join(input_dir, "y_test.npy")),
        "scaler": joblib.load(os.path.join(input_dir, "scaler.pkl")),
        "label_encoder": joblib.load(os.path.join(input_dir, "label_encoder.pkl")),
        "metadata": joblib.load(os.path.join(input_dir, "metadata.pkl")),
    }
    print(f"Loaded preprocessed data from {input_dir}/")
    return data


def train_model(X_train, y_train, model_params):
    """Train a RandomForestClassifier."""
    model = RandomForestClassifier(
        n_estimators=model_params["n_estimators"],
        max_depth=model_params["max_depth"],
        min_samples_split=model_params["min_samples_split"],
        min_samples_leaf=model_params["min_samples_leaf"],
        random_state=model_params["random_state"],
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print(f"Model trained: RandomForestClassifier with {model_params['n_estimators']} estimators")
    return model


def save_model_bundle(model, scaler, label_encoder, metadata, output_path="models/npk_crop_model.pkl"):
    """Save the complete model bundle for the Streamlit app."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    bundle = {
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_names": metadata["feature_names"],
        "target_names": metadata["target_names"],
        "accuracy": None,  # Will be set after evaluation
    }
    joblib.dump(bundle, output_path)
    print(f"Model bundle saved to {output_path}")
    return bundle


def main():
    """Run the full training pipeline with MLflow tracking."""
    params = load_params()
    model_params = params["model"]
    mlflow_config = params["mlflow"]

    # Set up MLflow
    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
    mlflow.set_experiment(mlflow_config["experiment_name"])

    # Load data
    data = load_preprocessed()

    with mlflow.start_run(run_name="rf-training"):
        # Log parameters
        mlflow.log_params({
            "n_estimators": model_params["n_estimators"],
            "max_depth": model_params["max_depth"],
            "min_samples_split": model_params["min_samples_split"],
            "min_samples_leaf": model_params["min_samples_leaf"],
            "test_size": params["data"]["test_size"],
        })

        # Train
        model = train_model(data["X_train"], data["y_train"], model_params)

        # Quick accuracy check
        train_acc = accuracy_score(data["y_train"], model.predict(data["X_train"]))
        test_acc = accuracy_score(data["y_test"], model.predict(data["X_test"]))

        mlflow.log_metrics({
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
        })

        print(f"Train accuracy: {train_acc:.4f}")
        print(f"Test accuracy:  {test_acc:.4f}")

        # Save model bundle
        bundle = save_model_bundle(
            model, data["scaler"], data["label_encoder"], data["metadata"]
        )
        bundle["accuracy"] = test_acc
        joblib.dump(bundle, "models/npk_crop_model.pkl")

        # Log model to MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")

        # Log the full bundle as artifact
        mlflow.log_artifact("models/npk_crop_model.pkl", "model_bundle")

        print(f"\n✅ Training complete! MLflow run logged.")
        print(f"   Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
