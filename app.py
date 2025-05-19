
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split

# Load the trained model
model = joblib.load("sleep_disorder_svm_model_v132.pkl")

st.set_page_config(page_title="Sleep Disorder Predictor", layout="centered")
st.title("🛌 Sleep Disorder Prediction App")
st.write("Enter your health and lifestyle information to predict potential sleep disorders.")

# User input fields
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 90, 30)
occupation = st.selectbox("Occupation", [
    "Doctor", "Nurse", "Engineer", "Teacher", "Lawyer", "Accountant",
    "Salesperson", "Software Engineer", "Scientist", "Manager", "Others"
])
bmi_category = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese"])
daily_steps = st.number_input("Daily Steps", min_value=0, max_value=30000, value=5000)

# Format input into DataFrame
input_data = [[gender, age, occupation, bmi_category, daily_steps]]
columns = ["Gender", "Age", "Occupation", "BMI Category", "Daily Steps"]
X_sample = pd.DataFrame(input_data, columns=columns)

# Predict and show result
if st.button("Predict"):
    prediction = model.predict(X_sample)[0]
    probas = model.predict_proba(X_sample)[0]

    st.success(f"🧠 Predicted Sleep Disorder: **{prediction}**")
    st.info(f"Prediction Probabilities:\nInsomnia: {probas[0]:.2f}, None: {probas[1]:.2f}, Sleep Apnea: {probas[2]:.2f}")

# Optional evaluation toggle
if st.checkbox("📊 Show model evaluation metrics (static)"):
    # Load dataset
    df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
    X = df[['Gender', 'Age', 'Occupation', 'BMI Category', 'Daily Steps']]
    y = df['Sleep Disorder']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = model.predict(X_test)

    # Convert to string to avoid value errors
    y_test = y_test.astype(str)
    y_pred = pd.Series(y_pred).astype(str)
    labels = sorted(list(set(y_test) | set(y_pred)))

    # Confusion matrix
    st.subheader("📉 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig_cm, ax_cm = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax_cm)
    st.pyplot(fig_cm)

    # Classification report bar chart
    st.subheader("📊 Classification Metrics")
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics = ['precision', 'recall', 'f1-score']
    metric_values = {metric: [report[c][metric] for c in labels] for metric in metrics}

    x = np.arange(len(labels))
    width = 0.25
    fig_bar, ax_bar = plt.subplots()
    for i, metric in enumerate(metrics):
        ax_bar.bar(x + i * width, metric_values[metric], width=width, label=metric)
    ax_bar.set_xticks(x + width)
    ax_bar.set_xticklabels(labels)
    ax_bar.set_ylim(0, 1.1)
    ax_bar.set_title("Precision / Recall / F1-Score")
    ax_bar.legend()
    st.pyplot(fig_bar)
