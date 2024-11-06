import base64
from io import BytesIO
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np
from flask_bootstrap import Bootstrap5
import json
import matplotlib.pyplot as plt
import random

app = Flask(__name__)
bootstrap = Bootstrap5(app)

model = joblib.load("C:/Users/lucar/Documents/Avans/Jaar 4/Minor AI/ML/HCAID/Travel AI/Bad_App/model/Biased_Travel_AI.pkl")
encoder = LabelEncoder()
encoder.classes_ = np.load("C:/Users/lucar/Documents/Avans/Jaar 4/Minor AI/ML/HCAID/Travel AI/Bad_App/model/classes.npy", allow_pickle=True)

# Define the mapping dictionaries
budget_map = {"Laag": 1, "Midden": 2, "Hoog": 3}
climate_map = {"Tropisch": 1, "Gematigd": 2, "Koud": 3}
activities_map = {"Avontuur": 1, "Sightseeing": 2, "Ontspannen": 3}
companions_map = {"Alleen": 1, "Koppel": 2, "Famillie": 3, "Vrienden": 4}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/feedback")
def feedback():
    return render_template("feedback.html")

@app.route("/over_de_ai")
def about():
    return render_template("about.html")

@app.route("/voorspel")
def prediction():
    return render_template("predict.html")

@app.route("/predict", methods=['POST'])
def predict():
    data = json.loads(request.data.decode("utf-8")).get('data')
    budget, climate, activities, companions = data.split(',')

    # Encode input data using predefined maps
    encoded_input = np.array([
        budget_map.get(budget.strip(), 0),
        climate_map.get(climate.strip(), 0),
        activities_map.get(activities.strip(), 0),
        companions_map.get(companions.strip(), 0),
    ], dtype=np.float32).reshape(1, -1)  # Ensure dtype is np.float32

    encoded_prediction = model.predict(encoded_input)
    prediction = encoder.inverse_transform(encoded_prediction)

    # Split the prediction
    split_prediction = prediction[0].split(" - ")

    destination = split_prediction[0]
    accommodation_type = split_prediction[1]

    # Generate prices based on accommodation type
    if accommodation_type == "Hotel":
        price = random.randint(250, 750)
        acc_img = "../static/Hotel.png"
    elif accommodation_type == "Resort":
        price = random.randint(1350, 2000)
        acc_img = "../static/Resort.png"
    else:  # hostel
        price = random.randint(50, 250)
        acc_img = "../static/Hostel.png"

    # Generate the card HTML for booking links
    booking_card_html = f"""
    <div class="card" style="width: 12rem;">
      <img src="{acc_img}" class="card-img-top" alt="destination image" style="width: 100%; height: 150px; object-fit: cover;">
      <div class="card-body">
        <h5 class="card-title">{destination}</h5>
        <p class="card-text">
            Was: <span style="text-decoration: line-through;">€{price}</span>
            <br />
            Nu: <span style="color: red; font-weight: bold;">€{price - 100}</span>
            <br />
            <span style="color: red; font-weight: bold;">Nog 2 kamers!</span>
        </p>
        <a href="https://booking/{destination.lower()}/{accommodation_type}" class="btn btn-primary" target="_blank">Boek nu!</a>
      </div>
    </div>
    """

    # Return the final HTML with the booking card, alternatives, and plot
    result_html = f"""
    <h4>Je droombestemming: <strong>{"Een " + accommodation_type + " in " + destination}</strong></h4>
    <div class="card-deck">
        {booking_card_html}
    </div>
    """

    return jsonify(prediction=result_html)

if __name__ == "__main__":
    app.run(port=5001, debug=True)