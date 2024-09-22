import pickle
import pandas as pd
import xgboost as xgb  # Add this import for DMatrix
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# base directory
BASE_DIR = Path(__file__).resolve(strict=True).parent

# load the actual model
with open(f"{BASE_DIR}/plantGrowthModel/plant_growth_model.pkl", "rb") as f:
    model = pickle.load(f)

# Label encoders for the categorical features
le_soil = LabelEncoder()
le_water = LabelEncoder()
le_fertilizer = LabelEncoder()

# Fit the label encoders with the training data categories
soil_categories = ['Clay', 'Loamy', 'Sandy']
water_categories = ['Daily', 'Weekly', 'Monthly']
fertilizer_categories = ['Organic', 'Inorganic', 'None']

le_soil.fit(soil_categories)
le_water.fit(water_categories)
le_fertilizer.fit(fertilizer_categories)

def preprocess_input(soil_type, water_frequency, fertilizer_type, sunlight_hours, temperature, humidity):
    """
    Preprocess the input data to match the format used during model training.
    This function will use label encoding for categorical features and add continuous features.
    """
    data = {
        "Soil_Type": le_soil.transform([soil_type])[0],
        "Sunlight_Hours": sunlight_hours,
        "Water_Frequency": le_water.transform([water_frequency])[0],
        "Fertilizer_Type": le_fertilizer.transform([fertilizer_type])[0],
        "Temperature": temperature,
        "Humidity": humidity
    }
    
    return pd.DataFrame([data])


def predict_pipeline(soil_type, water_frequency, fertilizer_type, sunlight_hours, temperature, humidity):
    """
    Preprocess the input and predict the plant growth stage.
    """
    # Add missing features
    input_data = preprocess_input(soil_type, water_frequency, fertilizer_type, sunlight_hours, temperature, humidity)
    
    # Convert the input data to DMatrix format for XGBoost
    dmatrix_data = xgb.DMatrix(input_data)
    
    # Predict using the model (returns a float probability)
    pred = model.predict(dmatrix_data)
    
    # Convert the prediction to a meaningful string
    binary_prediction = 1 if pred[0] > 0.5 else 0
    growth_stage = "Reached" if binary_prediction == 1 else "Not Reached"
    
    return growth_stage

