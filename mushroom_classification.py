import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc

# Load dataset
data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data", header=None)

# Define column names
columns = ["class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment", "gill-spacing",
           "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring",
           "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
           "ring-number", "ring-type", "spore-print-color", "population", "habitat"]
data.columns = columns

# Define full-form mappings
full_form_options = {
    "cap-shape": {"b": "Bell", "c": "Conical", "f": "Flat", "k": "Knobbed", "s": "Sunken", "x": "Convex"},
    "cap-surface": {"f": "Fibrous", "g": "Grooves", "s": "Smooth", "y": "Scaly"},
    "cap-color": {"b": "Buff", "c": "Cinnamon", "e": "Red", "g": "Gray", "n": "Brown", "p": "Pink", "r": "Purple", "u": "Blue", "w": "White", "y": "Yellow"},
    "bruises": {"f": "No", "t": "Yes"},
    "odor": {"a": "Almond", "c": "Creosote", "f": "Foul", "l": "Anise", "m": "Musty", "n": "None", "p": "Pungent", "s": "Spicy", "y": "Fishy"},
    "gill-attachment": {"a": "Attached", "f": "Free"},
    "gill-spacing": {"c": "Close", "w": "Wide"},
    "gill-size": {"b": "Broad", "n": "Narrow"},
    "gill-color": {"b": "Buff", "e": "Red", "g": "Gray", "h": "Chocolate", "k": "Black", "n": "Brown", "o": "Orange", "p": "Pink", "r": "Purple", "u": "Blue", "w": "White", "y": "Yellow"},
    "stalk-shape": {"e": "Enlarging", "t": "Tapering"},
    "stalk-root": {"b": "Bulbous", "c": "Club", "e": "Equal", "r": "Rooted", "?": "Unknown"},
    "stalk-surface-above-ring": {"f": "Fibrous", "k": "Silky", "s": "Smooth", "y": "Scaly"},
    "stalk-surface-below-ring": {"f": "Fibrous", "k": "Silky", "s": "Smooth", "y": "Scaly"},
    "stalk-color-above-ring": {"b": "Buff", "c": "Cinnamon", "e": "Red", "g": "Gray", "n": "Brown", "o": "Orange", "p": "Pink", "w": "White", "y": "Yellow"},
    "stalk-color-below-ring": {"b": "Buff", "c": "Cinnamon", "e": "Red", "g": "Gray", "n": "Brown", "o": "Orange", "p": "Pink", "w": "White", "y": "Yellow"},
    "veil-type": {"p": "Partial"},
    "veil-color": {"n": "Brown", "o": "Orange", "w": "White", "y": "Yellow"},
    "ring-number": {"n": "None", "o": "One", "t": "Two"},
    "ring-type": {"e": "Evanescent", "f": "Flaring", "l": "Large", "n": "None", "p": "Pendant"},
    "spore-print-color": {"b": "Buff", "h": "Chocolate", "k": "Black", "n": "Brown", "o": "Orange", "r": "Purple", "u": "Blue", "w": "White", "y": "Yellow"},
    "population": {"a": "Abundant", "c": "Clustered", "n": "Numerous", "s": "Scattered", "v": "Several", "y": "Solitary"},
    "habitat": {"d": "Woods", "g": "Grasses", "l": "Leaves", "m": "Meadows", "p": "Paths", "u": "Urban", "w": "Waste"}
}

# Encode categorical features
label_encoders = {}
categorical_options = {}
for col in data.columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le
    categorical_options[col] = [full_form_options[col][key] if col in full_form_options and key in full_form_options[col] else key for key in le.classes_]

# Split dataset
X = data.drop("class", axis=1)
y = data["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# Compute AUC-ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot AUC-ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Function to encode user input
def encode_user_input(sample, label_encoders, feature_names):
    encoded_sample = []
    for i, feature in enumerate(sample):
        if feature in label_encoders[feature_names[i]].classes_:
            encoded_sample.append(label_encoders[feature_names[i]].transform([feature])[0])
        else:
            encoded_sample.append(-1)  # Assign unknown values
    return np.array(encoded_sample).reshape(1, -1)

# Get user input
print("Enter mushroom characteristics:")
user_input = []
feature_names = X.columns.tolist()
for feature in feature_names:
    options = categorical_options[feature]
    print(f"Options for {feature}: {options}")
    value = input(f"Enter {feature}: ")
    while value not in options:
        print("Invalid input. Please choose from the given options.")
        value = input(f"Enter {feature}: ")
    user_input.append(value)

# Encode and predict user input
encoded_input = encode_user_input(user_input, label_encoders, feature_names)
prediction = model.predict(encoded_input)[0]
pred_result = "Edible" if prediction == 0 else "Poisonous"
print(f"Predicted: {pred_result}")
