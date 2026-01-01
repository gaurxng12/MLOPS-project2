# promote model (MLflow alias-based)

import os
import mlflow
from mlflow.tracking import MlflowClient


def promote_model():
    # Set up DagsHub credentials for MLflow tracking
    dagshub_token = os.getenv("CAPSTONE_TEST")
    if not dagshub_token:
        raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "bgaurangan"
    repo_name = "MLOPS-project2"

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

    client = MlflowClient()
    model_name = "my_model"

    # Get model version currently tagged as @staging
    try:
        staging_model = client.get_model_version_by_alias(model_name, "staging")
    except Exception:
        print("No model found with alias @staging. Skipping promotion.")
        return

    staging_version = staging_model.version

    # Promote staging model to production (alias-based)
    client.set_registered_model_alias(
        name=model_name,
        alias="production",
        version=staging_version
    )

    print(f"Model version {staging_version} promoted to @production")


if __name__ == "__main__":
    promote_model()
