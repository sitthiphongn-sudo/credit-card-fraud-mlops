import mlflow
from mlflow.tracking import MlflowClient


def register_champion_model(run_id: str):
    client = MlflowClient()
    model_name = "fraud-detector"
    model_uri = f"runs:/{run_id}/model"

    print(f"Registering model from run {run_id}...")
    
    model_version = mlflow.register_model(model_uri, model_name)

    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=model_version.version
    )
    print(f"Successfully registered {model_name} v{model_version.version} as @champion!")

if __name__ == "__main__":
    RUN_ID = "8b650f8ee0c74dc79a4c186c56c6e8e8"
    register_champion_model(RUN_ID)