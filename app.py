import streamlit as st
import pickle
import pandas as pd


# Load the trained model and encoders
model = pickle.load(open('/Users/praneethkumarkandukuri/Desktop/CollegePredictionApp/my_model.pkl', 'rb'))
encoder_branch = pickle.load(open('encoder_branch.pkl', 'rb'))
encoder_college = pickle.load(open('encoder_college.pkl', 'rb'))


# Function to make predictions
def predict_college(user_rank, branch_name, gender, caste):
    branch_encoded = encoder_branch.transform([branch_name])[0]
    combined_rank = user_rank
    user_input = [[combined_rank, branch_encoded]]
    predicted_college_encoded = model.predict(user_input)[0]
    predicted_college = encoder_college.inverse_transform([predicted_college_encoded])[0]
    return predicted_college


# Streamlit UI
st.title('College Prediction System')
st.write("Enter your details below to predict the college you'll be admitted to:")


# Inputs
user_rank = st.number_input("Enter Your Rank", min_value=0, step=1)
branch_name = st.selectbox("Select Your Branch", ['COMPUTER SCIENCE AND ENGINEERING', 'ELECTRICAL ENGINEERING', 'MECHANICAL ENGINEERING', 'CIVIL ENGINEERING'])  # Update with all branches
gender = st.selectbox("Select Your Gender", ['BOYS', 'GIRLS'])
caste = st.selectbox("Select Your Caste", ['BC_B', 'SC', 'ST', 'OC', 'EWS', 'BC_A', 'BC_C', 'BC_D', 'BC_E'])  # Update with caste options


# Prediction button
if st.button("Predict College"):
    predicted_college = predict_college(user_rank, branch_name, gender, caste)
    st.success(f"Predicted College: {predicted_college}")

