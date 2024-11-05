from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np
from flask_bootstrap import Bootstrap5
import json

app = Flask(__name__)

bootstrap = Bootstrap5(app)

# Load the trained model
model = joblib.load("C:/Users/lucar/Documents/Avans/Jaar 4/Minor AI/ML/HCAID/Travel AI/Good_App/model/NL_Travel_AI.pkl")

# Assume we also saved the encoder to reuse it for decoding predictions
encoder = LabelEncoder()
encoder.classes_ = np.load("C:/Users/lucar/Documents/Avans/Jaar 4/Minor AI/ML/HCAID/Travel AI/Good_App/model/classes.npy", allow_pickle=True)  # Save classes in training script

# Define the mapping dictionaries
budget_map = {"Low": 1, "Medium": 2, "High": 3}
climate_map = {"Tropical": 1, "Temperate": 2, "Cold": 3}
activities_map = {"Adventure": 1, "Sightseeing": 2, "Relaxation": 3}
companions_map = {"Solo": 1, "Couple": 2, "Family": 3, "Friends": 4}
language_map = {"Portuguese": 1, "Japanese": 2, "English": 3, "Local Language": 4}

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


@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from form
    data = json.loads(request.data.decode("utf-8")).get('data')

    # Parse input data; assume data comes in the order of: budget, climate, activities, companions, language, destination
    budget, climate, activities, companions, language = data.split(',')

    # Encode input data using predefined maps
    encoded_input = np.array([
        budget_map.get(budget.strip(), 0),    # Default to 0 if not found
        climate_map.get(climate.strip(), 0),
        activities_map.get(activities.strip(), 0),
        companions_map.get(companions.strip(), 0),
        # language_map.get(language.strip(), 0),
    ]).reshape(1, -1)

    print(encoded_input)

    # Predict the class
    encoded_prediction = model.predict(encoded_input)

    print(encoded_prediction)
    
    # Since you may want to decode the prediction, ensure your destination map is correctly structured
    # This assumes encoded_prediction corresponds directly to the destination mapping.
    prediction = encoder.inverse_transform(encoded_prediction)  # Get the destination name from the map

    # WIP?
    # booking_link = f"https://booking/{prediction.replace(' ', '-').lower()}/hotel"
    # alternative_links = [
    #     f"https://booking/{prediction.replace(' ', '-').lower()}/hostel",
    #     f"https://booking/{prediction.replace(' ', '-').lower()}/resort"
    # ]    

    # Example destination for demonstration
    booking_link = f"https://booking/temp/hotel"
    alternative_links = [
        f"https://booking/temp/hostel",
        f"https://booking/temp/resort"
    ]    

    # Create an HTML snippet to be displayed on the page
    result_html = f"""
    <p>Jouw droombestemming is: <strong>{prediction}</strong></p>
    <p>Boek nu: <a href="{booking_link}" target="_blank">{booking_link}</a></p>
    <p>Liever een andere accommodatie?</p>
    <ul>
        <li><a href="{alternative_links[0]}" target="_blank">{alternative_links[0]}</a></li>
        <li><a href="{alternative_links[1]}" target="_blank">{alternative_links[1]}</a></li>
    </ul>
    <p>Deze resultaten zijn gebaseerd op:</p>
    """

    return jsonify(prediction=result_html)

if __name__ == "__main__":
    app.run(debug=True)