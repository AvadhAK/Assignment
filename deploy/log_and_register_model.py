import mlflow
import os
from mlflow.tracking import MlflowClient

print("1")
# Set tracking URI
# mlflow.set_tracking_uri("http://127.0.0.1/5000")

# Define experiment name
experiment_name = 'my-experiment23'
mlflow.set_experiment(experiment_name)

print("2")
# Log model
with mlflow.start_run(run_name='model-deployment-run23') as run:
    parent_dir = os.path.dirname(os.getcwd())  # Get the parent directory
    model_path = os.path.join(parent_dir, "Assignment", "Project", "model.pkl")

    print(model_path)
    mlflow.log_artifact(model_path, artifact_path='model_artifacts')

    print("2")
    # Register model
    model_uri = f'runs:/{run.info.run_id}/model_artifacts/model.pkl'
    mlflow.register_model(model_uri, 'my-registered-modeL23')

print('Model registered successfully!')
