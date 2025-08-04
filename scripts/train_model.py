import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pickle
import os

# Define file paths
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'insurance.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'model.pkl')

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies feature engineering to the raw dataframe.
    This logic must be identical to the one used during inference.
    """
    # Create a copy to avoid modifying the original DataFrame
    df_feat = df.copy()

    # 1. BMI Calculation
    df_feat["bmi"] = df_feat["weight"] / (df_feat["height"] ** 2)

    # 2. Age Group Categorization
    def age_group(age):
        if age < 25: return "young"
        elif age < 45: return "adult"
        elif age < 60: return "middle_aged"
        return "senior"
    df_feat["age_group"] = df_feat["age"].apply(age_group)

    # 3. Lifestyle Risk Categorization
    def lifestyle_risk(row):
        if row["smoker"] and row["bmi"] > 30: return "high"
        elif row["smoker"] or row["bmi"] > 27: return "medium"
        else: return "low"
    df_feat["lifestyle_risk"] = df_feat.apply(lifestyle_risk, axis=1)

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
    def city_tier(city):
        if city in tier_1_cities: return 1
        elif city in tier_2_cities: return 2
        else: return 3
    df_feat["city_tier"] = df_feat["city"].apply(city_tier)

    # Drop original columns that were used for feature engineering
    df_feat = df_feat.drop(columns=['age', 'weight', 'height', 'smoker', 'city'])
    
    return df_feat

def train_and_save_model():
    """
    Main function to orchestrate the model training and saving process.
    """
    print("Starting model training process...")

    # 1. Load Data
    print(f"Loading data from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: The data file was not found at {DATA_PATH}")
        return

    # 2. Feature Engineering
    print("Performing feature engineering...")
    df_feat = feature_engineer(df)
    
    # 3. Define Features (X) and Target (y)
    X = df_feat.drop("insurance_premium_category", axis=1)
    y = df_feat["insurance_premium_category"]
    print("Features for training:", X.columns.tolist())

    # 4. Define the preprocessing pipeline
    # These are the features that will be one-hot encoded
    categorical_features = ["age_group", "lifestyle_risk", "occupation"]
    # These numerical features will be passed through without scaling
    numeric_features = ["bmi", "city_tier", "income_lpa"]

    # Create the column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ("num", "passthrough", numeric_features)
        ],
        remainder='drop' # Drop any columns not specified
    )

    # 5. Create the full model pipeline
    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42, n_estimators=100))
    ])

    # 6. Train the model on the entire dataset
    print("Training the model...")
    model_pipeline.fit(X, y)
    print("Model training completed.")

    # 7. Save the trained pipeline
    print(f"Saving model to {MODEL_PATH}...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True) # Ensure model directory exists
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_pipeline, f)
    print("Model saved successfully!")

if __name__ == "__main__":
    train_and_save_model()
