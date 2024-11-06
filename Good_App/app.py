from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np
from flask_bootstrap import Bootstrap5
import json

app = Flask(__name__)
bootstrap = Bootstrap5(app)

model = joblib.load("C:/Users/lucar/Documents/Avans/Jaar 4/Minor AI/ML/HCAID/Travel AI/Good_App/model/No_Language_Travel_AI.pkl")
encoder = LabelEncoder()
encoder.classes_ = np.load("C:/Users/lucar/Documents/Avans/Jaar 4/Minor AI/ML/HCAID/Travel AI/Good_App/model/classes.npy", allow_pickle=True)

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


    # Track decision path and calculate feature influences recursively
    tree = model.tree_
    node_indicator = tree.decision_path(encoded_input)
    feature_influence = np.zeros(encoded_input.shape[1])

    for node_index in node_indicator.indices:
        # If this node uses a feature to make a split
        if tree.feature[node_index] != -2:  # -2 indicates a leaf node
            feature_index = tree.feature[node_index]
            feature_influence[feature_index] += abs(tree.threshold[node_index] - encoded_input[0, feature_index])

    # Normalize feature influences to percentages
    total_influence = feature_influence.sum()
    percentage_influences = (feature_influence / total_influence * 100).round(2)
    feature_names = ['Budget', 'Klimaat', 'Activiteiten', 'Reisgenootschap']

    # Generate HTML table for decision path influence
    influence_table_html = "<table><tr><th>Factor</th><th>Impact</th></tr>"
    for feature, influence in zip(feature_names, percentage_influences):
        influence_table_html += f"<tr><td>{feature}</td><td>{influence}%</td></tr>"
    influence_table_html += "</table>"

    # Generate booking links
    split_prediction = prediction[0].split(" - ")

    if split_prediction[1] == "hotel":
        booking_link = f"https://booking/{split_prediction[0].lower()}/hotel"
        alternative_links = [
            f"https://booking/{split_prediction[0].lower()}/hostel",
            f"https://booking/{split_prediction[0].lower()}/resort"
        ]   
    elif split_prediction[1] == "resort":
        booking_link = f"https://booking/{split_prediction[0].lower()}/resort"
        alternative_links = [
            f"https://booking/{split_prediction[0].lower()}/hostel",
            f"https://booking/{split_prediction[0].lower()}/hotel"
        ]   
    else:
        booking_link = f"https://booking/{split_prediction[0].lower()}/hostel"
        alternative_links = [
            f"https://booking/{split_prediction[0].lower()}/hotel",
            f"https://booking/{split_prediction[0].lower()}/resort"
        ]   

    # Create HTML output
    result_html = f"""
    <p>Jouw droombestemming is: <strong>{prediction[0]}</strong></p>
    <p>Boek nu: <a href="{booking_link}" target="_blank">{booking_link}</a></p>
    <p>Liever een andere accommodatie?</p>
    <ul>
        <li><a href="{alternative_links[0]}" target="_blank">{alternative_links[0]}</a></li>
        <li><a href="{alternative_links[1]}" target="_blank">{alternative_links[1]}</a></li>
    </ul>
    <p>Invloed van elke input op het resultaat:</p>
    {influence_table_html}
    """

    return jsonify(prediction=result_html)

if __name__ == "__main__":
    app.run(debug=True)
