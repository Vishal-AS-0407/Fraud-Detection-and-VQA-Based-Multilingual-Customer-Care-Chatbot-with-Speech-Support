import streamlit as st
import pandas as pd
import joblib
import smtplib
import os
from dotenv import load_dotenv
import subprocess

# Load environment variables from a .env file
load_dotenv()

# Fraud Detection Class
class FraudDetectionTest:
    def __init__(self, model_path, preprocessor_path):
        """Initialize by loading the trained model and preprocessor."""
        self.model = joblib.load(model_path)  # Load the saved ensemble model
        self.preprocessor = joblib.load(preprocessor_path)  # Load the preprocessor

    def preprocess_input(self, input_data):
        """Preprocess the input data using the preprocessor."""
        return self.preprocessor.transform(input_data)

    def predict(self, input_data):
        """Predict the fraud status (1: fraud, 0: not fraud)."""
        input_data_processed = self.preprocess_input(input_data)
        prediction = self.model.predict(input_data_processed)
        return prediction

# Send Email Notification
def send_email():
    try:
        # Fetching environment variables for email credentials
        sender_email = "vishal.as.aa@gmail.com"  # Your email
        receiver_email = "vishalpersonal04@gmail.com"  # Receiver email
        password = "uwnrsyrywbimwytc"  # Use the App Password here

        # Construct the email content
        message = """\
        Subject: Fraud Alert - Payment Attempt Detected

        A payment attempt was detected as fraudulent. Please review the transaction details."""

        # Connect to Gmail SMTP server and send email
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message)

    except smtplib.SMTPAuthenticationError as e:
        st.write(f"SMTP Authentication Error: {e}")
    except smtplib.SMTPException as e:
        st.write(f"SMTP Error: {e}")
    except Exception as e:
        st.write(f"Error sending email: {e}")

# Run the second Streamlit app (final.py)
def open_final_app():
    subprocess.Popen(["streamlit", "run", "final.py"])

# Streamlit UI code
def main():
    st.title("Payment Fraud Detection")

    # Create user input form
    step = st.number_input('Step:', min_value=1.0, max_value=100.0, value=42.0, step=1.0)
    payment_type = st.selectbox('Payment Type:', ['CASH_IN', 'CASH_OUT', 'DEBIT', 'TRANSFER'])
    amount = st.number_input('Amount:', min_value=0.0, max_value=10000000.0, value=1917497.633, step=0.01)
    nameOrig = st.text_input('Origin Name (e.g., C1747543519)', value='C1510964909')
    oldbalanceOrg = st.number_input('Old Balance (Origin):', min_value=0.0, max_value=10000000.0, value=1917497.63, step=0.01)
    newbalanceOrig = st.number_input('New Balance (Origin):', min_value=0.0, max_value=10000000.0, value=0.0, step=0.01)
    nameDest = st.text_input('Destination Name (e.g., C216972347)', value='C580102695')
    oldbalanceDest = st.number_input('Old Balance (Destination):', value=173507.14, step=0.01)
    newbalanceDest = st.number_input('New Balance (Destination):', value=2091004.76, step=0.01)

    # Create a DataFrame for input
    input_data = pd.DataFrame([{
        'step': step,
        'type': payment_type,
        'amount': amount,
        'nameOrig': nameOrig,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'nameDest': nameDest,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest
    }])

    # Load model and preprocessor
    test = FraudDetectionTest(model_path='ensemble_model.pkl', preprocessor_path='preprocessor.pkl')

    # When the user clicks the 'Done' button
    if st.button('Done'):
        # Predict fraud status
        prediction = test.predict(input_data)

        if prediction[0] == 1:
            st.warning("Fraudulent payment detected!")

            # Show the features
            st.write("Payment Details:")
            st.write(input_data)

            # Ask the user if they made this payment
            user_response = st.radio('Did you make this payment?', ['Yes', 'No'])

            if user_response == 'No':
                # Send email to notify about fraudulent payment
                send_email()
                st.success("Fraud notification sent via email!")
        else:
            st.success("No fraud detected. Payment is safe.")

    # Button to open the second app (final.py)
    if st.button('Customer App'):
        open_final_app()

if __name__ == "__main__":
    main()
