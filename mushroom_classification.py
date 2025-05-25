# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Load dataset
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
    columns = ["class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment", "gill-spacing",
               "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring",
               "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type",
               "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"]
    data = pd.read_csv(url, header=None, names=columns)
    return data

data = load_data()

# Encode features
label_encoders = {}
for col in data.columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Split data
X = data.drop("class", axis=1)
y = data["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.title("🍄 Mushroom Edibility Classifier")
st.write("Select mushroom characteristics to predict if it’s edible or poisonous:")

user_input = []
for feature in X.columns:
    options = label_encoders[feature].classes_
    selection = st.selectbox(f"{feature}", options)
    encoded = label_encoders[feature].transform([selection])[0]
    user_input.append(encoded)

# Predict
if st.button("Predict"):
    input_array = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    label = "Edible" if prediction == 0 else "Poisonous"
    st.success(f"The predicted mushroom is **{label}**!")

    # ROC Curve
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    st.subheader("AUC-ROC Curve")
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="blue")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)
