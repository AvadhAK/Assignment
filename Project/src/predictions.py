import pickle
import numpy as np
import os


def load_model_and_predict(sample_data):
    # Load the saved model
    parent_dir = os.path.dirname(os.getcwd())  # Get the parent directory
    model_path = os.path.join(parent_dir, "Assignment", "Project", "model.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    # Predict using the model
    prediction = model.predict(sample_data)
    return prediction


if __name__ == "__main__":
    # Sample data for prediction (first row of Iris dataset)
    sample_data = np.array([[5.1, 3.5, 1.4, 0.2]])
    prediction = load_model_and_predict(sample_data)
    print(f"Prediction: {prediction}")
