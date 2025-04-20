import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the trained model and encoders
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

loaded_model = model_data["model"]
feature_names = model_data["features_names"]

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Streamlit app
st.title("Customer Churn Prediction")
st.write("This app predicts whether a customer will churn based on their details.")

# Input fields for user data
input_data = {}

input_data['gender'] = st.selectbox('Gender', ['Male', 'Female'])
input_data['SeniorCitizen'] = st.selectbox('Senior Citizen', [0, 1])
input_data['Partner'] = st.selectbox('Partner', ['Yes', 'No'])
input_data['Dependents'] = st.selectbox('Dependents', ['Yes', 'No'])
input_data['tenure'] = st.number_input('Tenure (in months)', min_value=0, max_value=100, value=1)
input_data['PhoneService'] = st.selectbox('Phone Service', ['Yes', 'No'])
input_data['MultipleLines'] = st.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])
input_data['InternetService'] = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
input_data['OnlineSecurity'] = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
input_data['OnlineBackup'] = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
input_data['DeviceProtection'] = st.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
input_data['TechSupport'] = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
input_data['StreamingTV'] = st.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'])
input_data['StreamingMovies'] = st.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service'])
input_data['Contract'] = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
input_data['PaperlessBilling'] = st.selectbox('Paperless Billing', ['Yes', 'No'])
input_data['PaymentMethod'] = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
input_data['MonthlyCharges'] = st.number_input('Monthly Charges', min_value=0.0, value=0.0)
input_data['TotalCharges'] = st.number_input('Total Charges', min_value=0.0, value=0.0)

# Convert input data to DataFrame
input_data_df = pd.DataFrame([input_data])

# Encode categorical features using the saved encoders
for column, encoder in encoders.items():
    input_data_df[column] = encoder.transform(input_data_df[column])

# Predict churn
if st.button("Predict Churn"):
    prediction = loaded_model.predict(input_data_df)
    pred_prob = loaded_model.predict_proba(input_data_df)

    st.write(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
    st.write(f"Prediction Probability: {pred_prob}")