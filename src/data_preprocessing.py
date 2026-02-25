"""
Data Preprocessing Module
- Loads CSV dataset
- Splits into train/test
- Applies StandardScaler
- Encodes labels
- Saves processed artifacts
"""

import os
import yaml
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_params(params_path="params.yaml"):
    """Load parameters from params.yaml."""
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def load_data(data_path):
    """Load the crop recommendation dataset."""
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape[0]} samples, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    return df


def preprocess(df, test_size=0.2, random_state=42):
    """
    Preprocess the dataset:
    - Feature/target split
    - Label encoding
    - Train/test split
    - Standard scaling
    """
    feature_cols = ["N", "P", "K"]
    target_col = "Crop"

    X = df[feature_cols].values
    y = df[target_col].values

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Train set: {X_train_scaled.shape[0]} samples")
    print(f"Test set:  {X_test_scaled.shape[0]} samples")
    print(f"Classes:   {list(label_encoder.classes_)}")

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_names": feature_cols,
        "target_names": list(label_encoder.classes_),
    }


def save_preprocessed(processed_data, output_dir="data/processed"):
    """Save preprocessed data artifacts."""
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "X_train.npy"), processed_data["X_train"])
    np.save(os.path.join(output_dir, "X_test.npy"), processed_data["X_test"])
    np.save(os.path.join(output_dir, "y_train.npy"), processed_data["y_train"])
    np.save(os.path.join(output_dir, "y_test.npy"), processed_data["y_test"])
    joblib.dump(processed_data["scaler"], os.path.join(output_dir, "scaler.pkl"))
    joblib.dump(processed_data["label_encoder"], os.path.join(output_dir, "label_encoder.pkl"))

    # Save metadata
    metadata = {
        "feature_names": processed_data["feature_names"],
        "target_names": processed_data["target_names"],
    }
    joblib.dump(metadata, os.path.join(output_dir, "metadata.pkl"))

    print(f"Preprocessed data saved to {output_dir}/")


def main():
    """Run the full preprocessing pipeline."""
    params = load_params()

    data_path = params["data"]["path"]
    test_size = params["data"]["test_size"]
    random_state = params["data"]["random_state"]

    df = load_data(data_path)
    processed = preprocess(df, test_size=test_size, random_state=random_state)
    save_preprocessed(processed)

    print("\n✅ Preprocessing complete!")


if __name__ == "__main__":
    main()
