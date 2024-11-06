from sklearn.preprocessing import LabelEncoder
import numpy as np

# Your original labels
original_labels = ['label1', 'label2', 'label3']  # Replace with your actual labels

# Initialize and fit the LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(original_labels)

# Print the LabelEncoder instance
print("LabelEncoder instance:", label_encoder)

# Print the classes and their corresponding encoded values
print("Classes (original labels):", label_encoder.classes_)

# Optionally, you can also print the mapping
for i, label in enumerate(label_encoder.classes_):
    print(f"Label: {label} -> Encoded value: {i}")
