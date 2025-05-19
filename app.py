
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Load model
model = joblib.load("sleep_disorder_svm_model.pkl")

# Page setup
st.set_page_config(page_title="Sleep Disorder Prediction", layout="centered")
st.title("🛌 Sleep Disorder Prediction App")
st.write("Provide the following details to predict possible sleep disorders:")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 90, 30)
occupation = st.selectbox("Occupation", ["Doctor", "Nurse", "Engineer", "Teacher", "Lawyer", "Accountant", "Salesperson", "Software Engineer", "Scientist", "Manager", "Others"])
bmi_category = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese"])
daily_steps = st.number_input("Daily Steps", min_value=0, max_value=30000, value=4000)

# Sample test data
X_test_sample = pd.DataFrame([{
    "Gender": gender,
    "Age": age,
    "Occupation": occupation,
    "BMI Category": bmi_category,
    "Daily Steps": daily_steps
}])

if st.button("Predict"):
    prediction = model.predict(X_test_sample)[0]
    probas = model.predict_proba(X_test_sample)[0]

    st.success(f"🧠 Predicted Sleep Disorder: **{prediction}**")
    st.info(f"Prediction Probabilities:\nInsomnia: {probas[0]:.2f}, None: {probas[1]:.2f}, Sleep Apnea: {probas[2]:.2f}")

    # Load full dataset to evaluate model (temporary for visualization)
    df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
    df = df[['Gender', 'Age', 'Occupation', 'BMI Category', 'Daily Steps', 'Sleep Disorder']]
    X = df.drop("Sleep Disorder", axis=1)
    y_true = df["Sleep Disorder"]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2, random_state=42)

    y_pred = model.predict(X_test)

    # Confusion Matrix
    st.subheader("📉 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    fig_cm, ax_cm = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(ax=ax_cm)
    st.pyplot(fig_cm)

    # Classification Report Bar Chart
    st.subheader("📊 Classification Metrics")
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    classes = list(model.classes_)
    metrics = ['precision', 'recall', 'f1-score']
    metric_values = {metric: [report_dict[c][metric] for c in classes] for metric in metrics}

    x = np.arange(len(classes))
    width = 0.25
    fig_bar, ax_bar = plt.subplots()
    for i, metric in enumerate(metrics):
        ax_bar.bar(x + i*width, metric_values[metric], width=width, label=metric)
    ax_bar.set_xticks(x + width)
    ax_bar.set_xticklabels(classes)
    ax_bar.set_ylim(0, 1.1)
    ax_bar.set_title("Precision / Recall / F1-Score")
    ax_bar.legend()
    st.pyplot(fig_bar)
