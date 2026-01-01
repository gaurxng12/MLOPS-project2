
import numpy as np
import pandas as pd
import pickle
import json
import yaml
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

import dagshub
from src.logger import logging


# -------------------------------------------------------
# MLflow + DagsHub setup
# -------------------------------------------------------
mlflow.set_tracking_uri("https://dagshub.com/bgaurangan/MLOPS-project2.mlflow")
dagshub.init(repo_owner="bgaurangan", repo_name="MLOPS-project2", mlflow=True)


# -------------------------------------------------------
# Utility functions
# -------------------------------------------------------
def load_params(file_path: str = "params.yaml") -> dict:
    try:
        with open(file_path, "r") as f:
            params = yaml.safe_load(f)
        logging.info("Parameters loaded from params.yaml")
        return params
    except FileNotFoundError:
        logging.warning("params.yaml not found, using defaults")
        return {
            "train_and_evaluate": {
                "C": 1.0,
                "solver": "liblinear",
                "penalty": "l2",
                "random_state": 42,
                "max_iter": 1000,
            }
        }


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    logging.info("Data loaded from %s", file_path)
    return df


def train_model(X, y, params) -> LogisticRegression:
    clf = LogisticRegression(**params)
    clf.fit(X, y)
    logging.info("Model training completed")
    return clf


def evaluate_model(clf, X, y) -> dict:
    y_pred = clf.predict(X)
    y_proba = clf.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "auc": roc_auc_score(y, y_proba),
    }

    logging.info("Model evaluation completed")
    return metrics


def save_metrics(metrics: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)


def save_model_locally(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    mlflow.set_experiment("my-dvc-pipeline2")

    with mlflow.start_run() as run:
        # Load data
        train_df = load_data("data/processed/train_bow.csv")
        test_df = load_data("data/processed/test_bow.csv")

        X_train = train_df.iloc[:, :-1].values
        y_train = train_df.iloc[:, -1].values
        X_test = test_df.iloc[:, :-1].values
        y_test = test_df.iloc[:, -1].values

        # Params
        params = load_params().get("train_and_evaluate")
        for k, v in params.items():
            mlflow.log_param(k, v)

        # Train
        clf = train_model(X_train, y_train, params)

        # Evaluate
        metrics = evaluate_model(clf, X_test, y_test)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        save_metrics(metrics, "reports/metrics.json")
        save_model_locally(clf, "models/model.pkl")

        # ---------------------------------------------------
        # Log + Register model (ONE STEP)
        # ---------------------------------------------------
        signature = infer_signature(X_test[:5], clf.predict(X_test[:5]))
        model_name = "sentiment_model"

        model_info = mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            signature=signature,
            input_example=X_test[:5],
            registered_model_name=model_name,
        )

        logging.info(f"Model registered: {model_info.model_uri}")

        # ---------------------------------------------------
        # Assign alias "staging"
        # ---------------------------------------------------
        client = MlflowClient()
        latest_version = client.get_latest_versions(model_name)[-1].version

        client.set_registered_model_alias(
            name=model_name,
            alias="staging",
            version=latest_version,
        )

        logging.info(
            f"Alias 'staging' assigned to {model_name} version {latest_version}"
        )

        # Save run info (optional, for audit)
        with open("reports/experiment_info.json", "w") as f:
            json.dump(
                {
                    "run_id": run.info.run_id,
                    "model_name": model_name,
                    "model_version": latest_version,
                    "alias": "staging",
                },
                f,
                indent=4,
            )

        mlflow.log_artifact("reports/metrics.json")

        print("\n" + "=" * 60)
        print("✅ TRAINING + REGISTRATION COMPLETE")
        print("=" * 60)
        print(f"Run ID: {run.info.run_id}")
        print(f"Model: {model_name}")
        print(f"Alias: staging")
        print("Metrics:")
        for k, v in metrics.items():
            print(f"  - {k}: {v:.4f}")
        print("\nLoad model with:")
        print("  mlflow.sklearn.load_model('models:/sentiment_model@staging')")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
