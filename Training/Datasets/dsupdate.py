import pandas as pd

# Load the CSV file
file_path = 'C:/Users/lucar/Documents/Avans/Jaar 4/Minor AI/ML/HCAID/Travel AI/Training/Datasets/destinations.csv'  # Replace with your file path if needed
data = pd.read_csv(file_path)

# Create the Language column based on the conditions
data['Language'] = data['Destination'].apply(
    lambda x: 'Portuguese' if 'Rio de Janeiro' in x else
              'Japanese' if 'Tokyo' in x else
              'English' if 'London' in x else
              'Local Language'
)

# Define the desired column order to place "Language" before "Destination"
# Extract all columns except "Language" and "Destination" in their original order
columns = list(data.columns)
columns.remove('Language')  # Temporarily remove 'Language' for controlled insertion
columns.remove('Destination')  # Temporarily remove 'Destination' to place it after 'Language'

# Insert "Language" before "Destination" and then append "Destination"
reordered_columns = columns + ['Language', 'Destination']

# Reorder the DataFrame
data = data[reordered_columns]

# Save the updated DataFrame back to a CSV file
output_file_path = 'updated_destinations.csv'  # Choose the output file path
data.to_csv(output_file_path, index=False)

print(f"Updated file saved to {output_file_path}")
