import mlflow
import numpy as np

# Load the model from the local MLflow server using the model name or ID
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# For a registered model, specify the name of the model in the format: 'models:/<model_name>/<version>'
model_name = "my-registered-model1"
model_version = "latest"  # Or use 'latest' to load the latest version of the model

model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.pyfunc.load_model(model_uri)

# Prepare the input data for inference (make sure it matches the expected format for your model)
input_data = np.array([[5.1,3.5,1.4,0.2]])  # Example input, modify based on your model's input

# Make a prediction using the model
prediction = model.predict(input_data)

# Output the prediction result
print("Model prediction:", prediction)
