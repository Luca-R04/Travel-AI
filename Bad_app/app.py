from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model and encoder
model = joblib.load("C:/Users/lucar/Documents/Avans/Jaar 4/Minor AI/ML/HCAID/Travel AI/Good_App/model/Biased_Travel_AI.pkl")

# Assume we also saved the encoder to reuse it for decoding predictions
encoder = LabelEncoder()
encoder.classes_ = np.load("C:/Users/lucar/Documents/Avans/Jaar 4/Minor AI/ML/HCAID/Travel AI/Bad_App/model/classes.npy", allow_pickle=True)  # Save classes in training script

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get input data from form
    data = request.form.get("input_data")
    # Convert to numpy array; replace this with your format of data entry
    input_data = np.array(data.split(',')).astype(float).reshape(1, -1)
    
    # Predict the encoded class
    encoded_prediction = model.predict(input_data)[0]
    
    # Decode to original class
    prediction = encoder.inverse_transform([encoded_prediction])[0]
    
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)