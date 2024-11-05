import pandas as pd

# Load the CSV file
file_path = 'C:/Users/lucar/Documents/Avans/Jaar 4/Minor AI/ML/HCAID/Travel AI/Training/Datasets/destinations.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Define the list of destinations to be labeled as "Resort"
resort_destinations = ["Kathmandu", "Tokyo", "Bali", "Rio de Janeiro", "Cape Town", "Maldives"]

# Function to modify the text after the hyphen
def update_accommodation(destination):
    place = destination.split(" - ")[0]  # Get the part before the hyphen
    if place in resort_destinations:
        return f"{place} - Resort"
    else:
        return f"{place} - Hotel"

# Apply the function to each row in the 'Destination' column
data['Destination'] = data['Destination'].apply(update_accommodation)

# Save the modified file
output_path = 'modified_travel_destinations.csv'
data.to_csv(output_path, index=False)

print(f"File saved as {output_path}")