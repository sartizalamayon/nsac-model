import pickle
import pandas as pd
import xgboost as xgb  # Add this import for DMatrix
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
#2nd Model
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


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



#2nd Model
# Define the SoilClassifier model (ResNet50 with custom classifier)
class SoilClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SoilClassifier, self).__init__()
        self.resnet = models.resnet50(weights=None)  # We are not using pretrained weights
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove the last fully connected layer

        # Custom classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.resnet(x)
        return self.classifier(features)

# Function to load model
def load_soil_model():
    num_classes = 8
    model = SoilClassifier(num_classes)
    state_dict = torch.load('app/model/soilClassifierModel/soil_type.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Preprocess the image
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resizing to match ResNet input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

# Predict soil type
def predict_soil_type(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Soil type labels
soil_types = ['Aluvial', 'Kapur', 'Andosol', 'Entisol', 'Laterit', 'Pasir', 'Humus', 'Inceptisol']

# Function to classify the soil
def classify_soil(image: Image.Image):
    model = load_soil_model()
    image_tensor = preprocess_image(image)
    predicted_index = predict_soil_type(model, image_tensor)
    predicted_soil_type = soil_types[predicted_index]
    return predicted_soil_type