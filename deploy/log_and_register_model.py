import mlflow
from mlflow.tracking import MlflowClient
import os

# Set tracking URI
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))

# Define experiment name
experiment_name = 'my-experiment'
mlflow.set_experiment(experiment_name)

# Log model
with mlflow.start_run(run_name='model-deployment-run') as run:
    parent_dir = os.path.dirname(os.getcwd())  # Get the parent directory
    model_path = os.path.join(parent_dir, "Assignment", "Project", "model.pkl")

    mlflow.log_artifact(model_path, artifact_path='model_artifacts')

    # Register model
    model_uri = f'runs:/{run.info.run_id}/model_artifacts/model.pkl'
    mlflow.register_model(model_uri, 'my-registered-model')

print('Model registered successfully!')
