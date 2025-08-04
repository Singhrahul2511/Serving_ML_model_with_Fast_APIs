Insurance Premium Prediction Service
Last Updated: August 05, 2025

1. Overview
This repository contains the complete source code for a machine learning-powered web service that predicts insurance premium categories (Low, Medium, High) based on user-provided data.

The project is architected as a modern, decoupled web service, comprising three distinct, independently functioning components:

Model Training Pipeline: A reproducible Python script that handles data ingestion, feature engineering, and model training, producing a serialized model artifact.

Backend API Service: A high-performance REST API built with FastAPI that exposes the trained model's prediction capabilities over a network.

Frontend Web Application: A user-friendly, interactive interface built with Streamlit that consumes the backend API.

This decoupled architecture ensures scalability, maintainability, and flexibility, allowing each component to be developed, tested, and deployed independently.

2. System Architecture & Data Flow
The application operates on a classic client-server model. The key principle is the separation of concerns: the frontend handles user interaction, while the backend handles all heavy computation and business logic.

The communication flow is as follows:

User Interaction (Frontend): A user accesses the Streamlit web application in their browser. They input their details (age, weight, city, etc.) into the form and click "Predict".

HTTP POST Request (Frontend → Backend):

The Streamlit application gathers the form data into a JSON object.

It sends this JSON payload via an HTTP POST request to the backend's /predict endpoint (e.g., http://127.0.0.1:8000/predict).

Request Processing (Backend):

The FastAPI server receives the incoming POST request.

Pydantic Validation: FastAPI automatically validates the incoming JSON against the UserInput Pydantic model. If the data is malformed (e.g., wrong data type, missing field), it immediately returns a 422 Unprocessable Entity error.

Feature Engineering: The validated data is passed to a dedicated function that performs the same feature engineering steps used during training (calculating BMI, determining age group, etc.).

Prediction: The resulting feature DataFrame is fed into the loaded Scikit-learn pipeline (model.pkl). The pipeline handles the one-hot encoding and passes the final vector to the Random Forest model to get a prediction.

HTTP JSON Response (Backend → Frontend):

The backend packages the prediction result into a clean JSON object (e.g., {"predicted_category": "Medium"}).

It sends this JSON back to the Streamlit application as the body of an HTTP 200 OK response.

Displaying Results (Frontend):

The Streamlit application receives the HTTP response.

It parses the JSON to extract the predicted category and dynamically updates the user interface to display the result in a success message.

This entire round-trip happens asynchronously in a matter of seconds.

3. Project Structure
The repository is organized into modules, each with a specific responsibility.

insurance_premium_predictor/
├── app/
│   ├── __init__.py
│   └── main.py             # FastAPI application: Handles API endpoints, validation, and prediction logic.
├── data/
│   └── insurance.csv       # Raw training data.
├── model/
│   └── model.pkl           # Serialized, trained Scikit-learn pipeline artifact.
├── scripts/
│   └── train_model.py      # Standalone script to train and serialize the model.
├── ui/
│   └── streamlit_app.py    # Streamlit application: Handles UI components and API calls.
├── .gitignore              # Specifies files and directories to be ignored by Git.
├── README.md               # This documentation file.
└── requirements.txt        # Python package dependencies for the entire project.

4. Getting Started
Prerequisites
Python 3.9+

pip and venv

Step-by-Step Installation
1. Clone the Repository & Create Environment

# Clone the repository from GitHub
git clone <your-repository-url>
cd insurance_premium_predictor

# Create a Python virtual environment
python -m venv venv

# Activate the environment
# On Windows:
# .\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

2. Install Dependencies
Install all required packages using the requirements.txt file.

pip install -r requirements.txt

3. Train the Machine Learning Model
This step is crucial. It runs the training script to generate the model.pkl file that the API relies on.

python scripts/train_model.py

You should see console output confirming that the model was trained and saved successfully to the model/ directory.

5. Running the Application
The backend and frontend must be run in two separate terminals.

Terminal 1: Launch the Backend API Service
This command starts the Uvicorn server, which serves your FastAPI application.

uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

--host: Binds the server to the local IP address.

--port: Makes the server available on port 8000.

--reload: Automatically restarts the server when code changes are detected (for development).

The API is now live and listening for requests at http://127.0.0.1:8000. You can view the auto-generated API documentation at http://127.0.0.1:8000/docs.

Terminal 2: Launch the Frontend Web Application
In a new terminal, activate the virtual environment again and run the Streamlit app.

streamlit run ui/streamlit_app.py

This command will automatically open a new tab in your default web browser, pointing to the Streamlit application. You can now interact with the UI to send requests to your running backend.