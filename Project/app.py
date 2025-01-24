import pickle
from flask import Flask, request, jsonify
import os
import numpy as np

# Load the saved model from the parent directory
parent_dir = os.path.dirname(os.getcwd())  # Get the parent directory
model_path = os.path.join(parent_dir, "Project", "model.pkl")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from request (assuming JSON format)
    data = request.json
    
    # Make prediction
    prediction = model.predict(np.array(data['input']).reshape(1, -1))
    
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
