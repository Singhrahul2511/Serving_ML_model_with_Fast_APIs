import pandas as pd
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import os

# --- 1. SETUP ---
# Get the absolute path of the current file
current_file_path = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the model file
MODEL_PATH = os.path.join(current_file_path, '..', 'model', 'model.pkl')

# Initialize the FastAPI app
app = FastAPI(
    title="Insurance Premium Prediction API",
    description="An API to predict insurance premium categories using a trained ML model.",
    version="1.0.0"
)

# Load the trained model pipeline
try:
    with open(MODEL_PATH, "rb") as f:
        model_pipeline = pickle.load(f)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    model_pipeline = None
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    model_pipeline = None


# --- 2. PYDANTIC MODELS ---
# Define the input data model for request validation
class UserInput(BaseModel):
    age: int = Field(..., gt=0, lt=120, example=35, description="Age of the user")
    weight: float = Field(..., gt=0, example=75.5, description="Weight of the user in kg")
    height: float = Field(..., gt=0, lt=2.5, example=1.75, description="Height of the user in meters")
    income_lpa: float = Field(..., gt=0, example=12.0, description="Annual salary in Lakhs Per Annum")
    smoker: bool = Field(..., example=False, description="Is the user a smoker?")
    city: str = Field(..., example="Mumbai", description="The city where the user resides")
    occupation: Literal[
        'retired', 'freelancer', 'student', 'government_job',
        'business_owner', 'unemployed', 'private_job'
    ] = Field(..., example='private_job', description="Occupation of the user")

# Define the output data model
class PredictionOutput(BaseModel):
    predicted_category: str


# --- 3. FEATURE ENGINEERING LOGIC ---
# This logic must be identical to the one in `scripts/train_model.py`
def feature_engineer_api(data: UserInput) -> pd.DataFrame:
    """Creates the feature DataFrame for a single API prediction."""
    
    # 1. BMI Calculation
    bmi = data.weight / (data.height ** 2)

    # 2. Age Group Categorization
    if data.age < 25: age_group = "young"
    elif data.age < 45: age_group = "adult"
    elif data.age < 60: age_group = "middle_aged"
    else: age_group = "senior"

    # 3. Lifestyle Risk Categorization
    if data.smoker and bmi > 30: lifestyle_risk = "high"
    elif data.smoker or bmi > 27: lifestyle_risk = "medium"
    else: lifestyle_risk = "low"

    # 4. City Tier Categorization
    tier_1_cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"]
    tier_2_cities = [
        "Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna", "Ranchi", "Visakhapatnam", "Coimbatore",
        "Bhopal", "Nagpur", "Vadodara", "Surat", "Rajkot", "Jodhpur", "Raipur", "Amritsar", "Varanasi",
        "Agra", "Dehradun", "Mysore", "Jabalpur", "Guwahati", "Thiruvananthapuram", "Ludhiana", "Nashik",
        "Allahabad", "Udaipur", "Aurangabad", "Hubli", "Belgaum", "Salem", "Vijayawada", "Tiruchirappalli",
        "Bhavnagar", "Gwalior", "Dhanbad", "Bareilly", "Aligarh", "Gaya", "Kozhikode", "Warangal",
        "Kolhapur", "Bilaspur", "Jalandhar", "Noida", "Guntur", "Asansol", "Siliguri"
    ]
    if data.city in tier_1_cities: city_tier = 1
    elif data.city in tier_2_cities: city_tier = 2
    else: city_tier = 3
    
    # Create a dictionary with the engineered features
    feature_dict = {
        "bmi": bmi,
        "age_group": age_group,
        "lifestyle_risk": lifestyle_risk,
        "city_tier": city_tier,
        "income_lpa": data.income_lpa,
        "occupation": data.occupation
    }
    
    # Convert to a DataFrame
    return pd.DataFrame([feature_dict])


# --- 4. API ENDPOINTS ---
@app.get("/", tags=["Status"])
def read_root():
    """Root endpoint to check API status."""
    return {"status": "API is running"}

@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict(user_input: UserInput):
    """
    Predicts the insurance premium category based on user input.
    """
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not available")

    try:
        # 1. Perform feature engineering on the input data
        features_df = feature_engineer_api(user_input)

        # 2. Make a prediction using the loaded pipeline
        # The pipeline handles both preprocessing and prediction
        prediction = model_pipeline.predict(features_df)
        
        # 3. Return the result
        return PredictionOutput(predicted_category=prediction[0])

    except Exception as e:
        # For any other errors during prediction
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")
