# 💳 Payment Fraud Detection and Customer Care Chatbot with Speech and Multilingual Support

This project consists of two primary components:

1. **Federated Learning-Based Payment Fraud Detection**  
   A system built using federated learning to detect fraudulent transactions in payment systems.

2. **VQA Customer Care Chatbot**  
   A multilingual, speech-to-text-enabled chatbot designed to assist customers with payment issues, form filling, and error resolution.

---

## 🔧 Requirements

To run this project, install the necessary dependencies via `requirements.txt`:

```bash
pip install -r requirements.txt
```


---

## 📂 Project Structure

```

├── train.py               # Federated Learning for Fraud Detection
├── test.py                # Fraud Detection Streamlit App
├── final.py               # VQA Customer Care Chatbot with Speech and Multilingual Support

```

---

## 📝 Scripts Overview

### 1. **`train.py`** – Federated Learning for Fraud Detection

This script handles the training of fraud detection models. It preprocesses the data, applies **SMOTE** for class imbalance, and trains both a **Neural Network** and an **Ensemble Random Forest Model**. 

#### Features:
- **Federated Learning Setup**: Combines models from different datasets using ensemble techniques.
- **SMOTE**: Balances the dataset to handle fraud class imbalance.
- **Model Saving**: Saves the trained models and preprocessing pipeline for future use.

---

### 2. **`test.py`** – Fraud Detection Streamlit Application

A web-based app to check if a payment transaction is fraudulent. If fraud is detected, it prompts the user for confirmation. If not, it sends an email alert to the bank manager.

#### Features:
- **Real-time Fraud Prediction**: Input transaction details to check for fraud.
- **User Confirmation**: Confirms if the user made the payment.
- **Bank Alert**: Sends email to the bank manager if the transaction is fraudulent.

---

### 3. **`final.py`** – VQA Customer Care Chatbot with Speech and Multilingual Support

A **Voice Query Answering (VQA)** system that assists customers with payment-related queries, form filling, and error resolution. It uses **Sarvam API** for **speech-to-text** conversion, supporting multiple Indian languages.

#### Features:
- **Speech-to-Text**: Converts spoken language to text using the Sarvam API.
- **Multilingual Support**: Supports multiple major Indian languages.
- **Image Upload**: Allows users to upload images for processing.
- **Privacy**: Ensures user details remain confidential.

---

## 📄 How to Run the Project

### 1. **Training the Fraud Detection Models**

Train the models by running:

```bash
python train.py
```

This will preprocess the data, apply SMOTE, and train the models.

---

### 2. **Running the Fraud Detection Web App**

After training, run the fraud detection app with Streamlit:

```bash
streamlit run test.py
```

This will open the app in your browser to check and confirm fraudulent transactions.

---

### 3. **Running the Customer Care Chatbot**

Start the chatbot with:

```bash
streamlit run final.py
```
---
### 🌟 **Show Your Support!**  

If you find this project helpful, give it a ⭐ on GitHub and share it with others! 😊  


