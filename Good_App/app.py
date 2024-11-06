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

    # Split the prediction
    split_prediction = prediction[0].split(" - ")

    destination = split_prediction[0]
    accommodation_type = split_prediction[1]

    # Generate prices based on accommodation type
    if accommodation_type == "Hotel":
        price = random.randint(250, 750)
        acc_img = "../static/Hotel.png"
    elif accommodation_type == "Resort":
        price = random.randint(750, 2000)
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
        <p class="card-text">Price: €{price}</p>
        <a href="https://booking/{destination.lower()}/{accommodation_type}" class="btn btn-primary" target="_blank">Boek {accommodation_type.capitalize()}</a>
      </div>
    </div>
    """

    # Generate alternative accommodation cards
    alternative_cards_html = "<div class='row'>"
    accommodation_types = ["hotel", "resort", "hostel"]

    for alt_type in accommodation_types:
        # Generate price for alternative accommodations
        if alt_type == "hotel":
            alt_price = random.randint(250, 750)
            alt_acc_img = "../static/Hotel.png"
        elif alt_type == "resort":
            alt_price = random.randint(750, 2000)
            alt_acc_img = "../static/Resort.png"
        else:  # hostel
            alt_price = random.randint(50, 250)
            alt_acc_img = "../static/Hostel.png"

        alternative_cards_html += f"""
        <div class="col-md-4 mb-3">
          <div class="card">
            <img src="{alt_acc_img}" class="card-img-top" alt="destination image" style="width: 100%; height: 100px; object-fit: cover;">
            <div class="card-body">
              <h5 class="card-title">{destination}</h5>
              <p class="card-text">Price: €{alt_price}</p>
              <a href="https://booking/{destination.lower()}/{alt_type}" class="btn btn-secondary" target="_blank">Boek {alt_type.capitalize()}</a>
            </div>
          </div>
        </div>
        """
    alternative_cards_html += "</div>"

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
    feature_names = ['Budget', 'Klimaat', 'Activiteit', 'Genootschap']

    # Generate HTML table for decision path influence
    influence_table_html = "<table><tr><th>Factor</th><th>Impact</th></tr>"
    for feature, influence in zip(feature_names, percentage_influences):
        influence_table_html += f"<tr><td>{feature}</td><td>{influence}%</td></tr>"
    influence_table_html += "</table>"

    # Generate bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#FFF9D9') 
    ax.set_facecolor('#FFF9D9')
    ax.bar(feature_names, percentage_influences, color='#3498db')
    ax.set_xlabel('Percentage Invloed (%)')
    ax.set_title('Invloed op voorspelling')

    # Convert the plot to a base64 string
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img_str = base64.b64encode(img_buf.read()).decode('utf-8')
    img_buf.close()

    # Embed the image in HTML
    chart_html = f'<img src="data:image/png;base64,{img_str}" alt="Invloed op voorspelling"/>'

    # Return the final HTML with the booking card, alternatives, and plot
    result_html = f"""
    <h4>Je droombestemming: <strong>{"Een " + accommodation_type + " in " + destination}</strong></h4>
    <div class="card-deck">
        {booking_card_html}
    </div>
    <h5>Toch een andere accomodatie?</h5>
    <div class="card-deck">
        {alternative_cards_html}
    </div>
    <p>De onderstaande grafiek laat zien hoeveel invloed elk antwoord gehad heeft op de uitkomst.
    <br />
    Houd er rekening mee dat deze percentages beïnvloedt kunnen worden door de nauwkeurigheid van het model. 
    <br />
    Sommige factoren kunnen aan het begin van het voorspellen al invloedrijk zijn maar later minder gebruikt worden.<p>
    {chart_html}
    """

    return jsonify(prediction=result_html)

if __name__ == "__main__":
    app.run(debug=True)