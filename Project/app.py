import pickle
from flask import Flask, request, jsonify
import os

# Load the saved model from the parent directory
parent_dir = os.path.dirname(os.getcwd())  # Get the parent directory
model_path = os.path.join(parent_dir+"/Assignment/Project/", "model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from request (assuming JSON format)
    data = request.get_json(force=True)
    # Extract features from the input data (for example, assuming it's a list)
    features = data['features']
    
    # Make prediction
    prediction = model.predict([features])
    
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
