import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open("logistic_regression_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the scaler
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit UI
st.title("🚢 Titanic Survival Prediction App")
st.write("Enter passenger details to predict survival")

# User inputs
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
age = st.number_input("Age", min_value=1, max_value=100, value=25)
sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=8, value=0)
parch = st.number_input("Parents/Children Aboard (Parch)", min_value=0, max_value=6, value=0)
fare = st.number_input("Fare", min_value=0.0, value=50.0)
sex_male = st.radio("Sex", ["Male", "Female"]) == "Male"
embarked = st.selectbox("Port of Embarkation", ['C', 'Q', 'S'])
embarked_Q = 1 if embarked == 'Q' else 0
embarked_S = 1 if embarked == 'S' else 0

# Prediction
if st.button("Predict Survival"):
    input_data = np.array([[pclass, age, sibsp, parch, fare, int(sex_male), embarked_Q, embarked_S]])
    input_data = scaler.transform(input_data)  # Normalize input
    prediction = model.predict(input_data)[0]
    
    # Display result
    st.write("### 🟢 Survived" if prediction == 1 else "### 🔴 Did Not Survive")
