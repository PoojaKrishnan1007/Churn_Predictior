import pandas as pd
import streamlit as st
import joblib
import numpy as np
import os

model_path = os.path.join(os.path.dirname(__file__), 'Churn_prediction_model')
model = joblib.load(model_path)
# Reverse encoding dictionaries
geography_mapping = {0: "France", 1: "Spain", 2: "Germany"}
gender_mapping = {0: "Female", 1: "Male"}

def main():
    st.title("Churn Predictor Model")
    # Create input fields for features
    CreditScore = st.number_input("Credit Score")
    Geography = st.number_input("Geography")
    Gender = st.number_input("Gender")
    Age = st.number_input("Age")
    Tenure = st.number_input("Tenure")
    Balance = st.number_input("Balance")
    NumOfProducts = st.number_input("Number of Products")
    HasCrCard = st.number_input("Has Credit Card")
    IsActiveMember = st.number_input("Is Active Member")
    EstimatedSalary = st.number_input("Estimated Salary")
        
    # Create a dataframe from the input values
    input_data = pd.DataFrame({
    'CreditScore': [CreditScore],  
    'Geography': [Geography], 
    'Gender': [Gender],  
    'Age': [Age],
    'Tenure': [Tenure],
    'Balance' : [Balance],
    'NumOfProducts': [NumOfProducts], 
    'HasCrCard': [HasCrCard], 
    'IsActiveMember': [IsActiveMember],
    'EstimatedSalary': [EstimatedSalary] 
        
    })
    if st.button("Predict"):
        with st.spinner('Calculating...'):  # Display a spinner while predicting
            prediction = model.predict(input_data)
            st.success(f"Predicted churn: {prediction[0]:,.2f}")  # Access the prediction value correctly
            st.balloons()

# Run the app
if __name__ == "__main__":
  main()
