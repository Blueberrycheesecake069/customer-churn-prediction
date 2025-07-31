import streamlit as st
import pandas as pd
import joblib

# --- 1. Load the Model and Pre-computation Artifacts ---

# Load the best performing model (let's assume it's Random Forest)
try:
    model = joblib.load('models/logistic_regression_model.joblib')
except FileNotFoundError:
    st.error("Model file not found. Please run the training script first to create the model.")
    st.stop()

# Load the scaler that was used to train the model
try:
    scaler = joblib.load('models/scaler.joblib')
except FileNotFoundError:
    st.error("Scaler file not found. Please modify your training script to save the scaler and run it again.")
    st.stop()
    
# Load the column order from the training data
try:
    train_columns = joblib.load('models/train_columns.joblib')
except FileNotFoundError:
    st.error("Column list file not found. Please modify your training script to save the training columns and run it again.")
    st.stop()


# --- 2. Define the User Interface ---

st.title('Customer Churn Prediction App 🔮')
st.write('Enter the customer details below to predict whether they will churn.')

st.sidebar.header('Customer Input Features')

monthly_charges = st.sidebar.slider('Monthly Charges ($)', 18.0, 120.0, 70.0)
total_charges = st.sidebar.slider('Total Charges ($)', 18.0, 9000.0, 2000.0)
num_additional_services = st.sidebar.slider('Number of Additional Services', 0, 6, 2)

contract = st.sidebar.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
internet_service = st.sidebar.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
payment_method = st.sidebar.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

paperless_billing = st.sidebar.radio('Paperless Billing', ['Yes', 'No'])
partner = st.sidebar.radio('Has a Partner?', ['Yes', 'No'])
dependents = st.sidebar.radio('Has Dependents?', ['Yes', 'No'])


# --- 3. Preprocess User Input and Make Prediction ---

def predict_churn():
    # Create a dictionary from the user input
    input_data = {
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'NumAdditionalServices': num_additional_services,
        'Contract_One year': 1 if contract == 'One year' else 0,
        'Contract_Two year': 1 if contract == 'Two year' else 0,
        'InternetService_Fiber optic': 1 if internet_service == 'Fiber optic' else 0,
        'InternetService_No': 1 if internet_service == 'No' else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment_method == 'Credit card (automatic)' else 0,
        'PaymentMethod_Electronic check': 1 if payment_method == 'Electronic check' else 0,
        'PaymentMethod_Mailed check': 1 if payment_method == 'Mailed check' else 0,
        'PaperlessBilling_Yes': 1 if paperless_billing == 'Yes' else 0,
        'Partner_Yes': 1 if partner == 'Yes' else 0,
        'Dependents_Yes': 1 if dependents == 'Yes' else 0,
    }

    input_df = pd.DataFrame([input_data])
    
    # Create a full dataframe with all the training columns and fill with zeros
    full_df = pd.DataFrame(columns=train_columns).fillna(0)

    # *** THIS IS THE CORRECTED LINE ***
    # Use pd.concat to combine the empty dataframe with the user input dataframe
    combined_df = pd.concat([full_df, input_df], ignore_index=True).fillna(0)
    
    # Reorder to match training columns
    final_df = combined_df[train_columns]

    # Scale the data using the loaded scaler
    scaled_df = scaler.transform(final_df)

    # Make a prediction
    prediction = model.predict(scaled_df)
    prediction_proba = model.predict_proba(scaled_df)

    return prediction, prediction_proba

# --- 4. Display the Prediction ---

if st.sidebar.button('Predict Churn'):
    prediction, prediction_proba = predict_churn()

    st.subheader('Prediction Result')
    if prediction[0] == 1:
        st.warning(f'**This customer is likely to CHURN.**')
    else:
        st.success(f'**This customer is likely to STAY.**')

    st.write('**Prediction Probability:**')
    st.write(f"Probability of Not Churning: {prediction_proba[0][0]:.2f}")
    st.write(f"Probability of Churning: {prediction_proba[0][1]:.2f}")
