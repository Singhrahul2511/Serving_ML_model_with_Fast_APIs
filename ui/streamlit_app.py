import streamlit as st
import requests
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="Insurance Premium Predictor",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- API URL ---
# This is the endpoint for your locally running FastAPI server
API_URL = "http://127.0.0.1:8000/predict"

# --- UI Styling ---
st.markdown("""
<style>
    /* General body styling */
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* Button styling */
    .stButton>button {
        background-color: #0068c9;
        color: white;
        border-radius: 12px;
        padding: 12px 28px;
        border: none;
        font-size: 18px;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #00509e;
    }
    /* Success message styling */
    .stSuccess {
        background-color: #e7f3ff;
        border-left: 6px solid #0068c9;
        padding: 1rem;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: bold;
    }
    /* Header styling */
    h1, h2, h3 {
        color: #00509e;
    }
</style>
""", unsafe_allow_html=True)


# --- Main Application ---
st.title("🛡️ Insurance Premium Predictor")
st.markdown(
    "Welcome! This tool uses a machine learning model to estimate your insurance premium category. "
    "Please provide your details below."
)

# --- Input Form ---
with st.form("prediction_form"):
    st.header("Enter Your Details")
    
    # Use columns for a more compact and organized layout
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35, help="Your current age in years.")
        weight = st.number_input("Weight (kg)", min_value=30.0, value=75.0, format="%.1f", help="Your weight in kilograms.")
        height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.75, format="%.2f", help="Your height in meters.")

    with col2:
        income_lpa = st.number_input("Annual Income (LPA)", min_value=0.1, value=10.0, format="%.1f", help="Your annual income in Lakhs Per Annum.")
        smoker = st.selectbox("Are you a smoker?", options=[False, True], format_func=lambda x: "Yes" if x else "No", help="Do you smoke regularly?")
        occupation = st.selectbox(
            "Occupation",
            ['private_job', 'government_job', 'business_owner', 'freelancer', 'student', 'retired', 'unemployed'],
            help="Select your primary occupation."
        )

    city = st.text_input("City", value="Mumbai", help="The city you reside in (e.g., Mumbai, Delhi).")

    # Submit button for the form
    submitted = st.form_submit_button("Predict My Premium")


# --- Prediction Logic ---
if submitted:
    # Prepare the data in the format expected by the API
    input_data = {
        "age": age,
        "weight": weight,
        "height": height,
        "income_lpa": income_lpa,
        "smoker": smoker,
        "city": city,
        "occupation": occupation
    }

    # Display a spinner while waiting for the API response
    with st.spinner('Analyzing your details and making a prediction...'):
        try:
            # Send the request to the FastAPI server
            response = requests.post(API_URL, json=input_data, timeout=10)
            
            # Check if the request was successful
            if response.status_code == 200:
                result = response.json()
                prediction = result["predicted_category"]
                st.success(f"Predicted Insurance Premium Category: **{prediction}**")
            else:
                # Handle API errors gracefully
                st.error(f"API Error (Status Code: {response.status_code})")
                try:
                    st.json(response.json())
                except json.JSONDecodeError:
                    st.text(f"Raw error response: {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("Connection Error: Could not connect to the prediction server. Please ensure the FastAPI server is running.")
        except requests.exceptions.Timeout:
            st.error("Request Timed Out: The server took too long to respond. Please try again later.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

st.markdown("---")
st.markdown("_Disclaimer: This prediction is for educational purposes only and is based on a limited dataset. It should not be considered financial advice._")
