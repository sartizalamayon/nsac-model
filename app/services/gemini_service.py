import google.generativeai as genai
# from dotenv import load_dotenv
# import os

# load_dotenv()

# Load API key from environment variables
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
GEMINI_API_KEY = "AIzaSyC_AcLdevWEfgbLPYSe3D58kae1pD-16Rc"
genai.configure(api_key=GEMINI_API_KEY)

def get_gemini_response(datas):
    """Function to call the Gemini API and get a personalized farming suggestion."""
    try:
        # Initialize the model
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Craft the prompt using farmer's data
        prompt = f"""
        Provide a personalized farming suggestion based on the following data:
        - Soil type: {datas['soil_type']}
        - Water frequency: {datas['water_frequency']} times per week
        - Fertilizer type: {datas['fertilizer_type']}
        - Sunlight hours: {datas['sunlight_hours']} hours per day
        - Temperature: {datas['temperature']}°C
        - Humidity: {datas['humidity']}%
        - Growth stage: {datas['growth_stage']}.
        
        Please keep the suggestion short, practical, and specific to the current conditions.
        """
        # Call the model to generate content based on the prompt
        response = model.generate_content(prompt)
        clean_response = response.text.replace('**', '').replace('\n', ' ').strip()
        # Return the generated text response
        return clean_response
    
    except Exception as e:
        print(f"Error occurred: {e}")
        return None
    
def get_weather_suggestion(data):
    try:
        # Initialize the model
        model = genai.GenerativeModel("gemini-1.5-flash")

        # Craft the prompt using farmer's data
        prompt = f"""
Provide a personalized farming suggestion based on the following weather data:
- Temperature: {data['temperature']}°C
- Humidity: {data['humidity']}%
- Air Condition Index:
    - PM2.5: {data['air_condition_index'][0]} µg/m³
    - SO2: {data['air_condition_index'][1]} µg/m³
    - NO2: {data['air_condition_index'][2]} µg/m³
    - O3: {data['air_condition_index'][3]} µg/m³
- Air Pressure: {data['air_pressure']} hPa
- Wind Speed: {data['wind_speed']} km/h
keep it between 50-70 words. Give it as plain text
"""


        # Call the model to generate content based on the prompt
        response = model.generate_content(prompt)
        if response.text:
            clean_response = response.text.replace('**', '').replace('\n', ' ').strip()
            return clean_response
        else:
            raise Exception("The response from the model was empty.")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        return "Could not generate a suggestion at this time."


import json

def get_weather_suggestion_json(data):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")

        # Craft the prompt using farmer's data
        prompt = f"""
        Provide a detailed response based on the following weather data for {data['crop']}:
        - Temperature: {data['temperature']}°C
        - Humidity: {data['humidity']}%
        - Air Condition Index:
            - PM2.5: {data['air_condition_index'][0]} µg/m³
            - SO2: {data['air_condition_index'][1]} µg/m³
            - NO2: {data['air_condition_index'][2]} µg/m³
            - O3: {data['air_condition_index'][3]} µg/m³
        - Air Pressure: {data['air_pressure']} hPa
        - Wind Speed: {data['wind_speed']} km/h

        Return the response as a plain text with specific advice for each parameter:
        - Temperature
        - Humidity
        - Air Quality (Air Condition Index)
        - Air Pressure
        - Wind Speed

        Give the response in a plain text separed by '|'. 
        For example -> '28.5°C is within the optimal range for corn growth (25-30°C). This temperature supports healthy photosynthesis and development| 65.0% humidity is also favorable for corn. This level provides sufficient moisture for plant growth without excessive stress| The provided Air Condition Index data indicates a moderate level of air pollution, 1015.0 hPa is within the normal range for atmospheric pressure.  This has no direct impact on corn growth.| 12.0 km/h is a moderate wind speed. While it might not directly affect corn growth, strong winds can cause stress to corn plants and potentially damage them. '
        Follow this format strictly
        """

        response = model.generate_content(prompt)
        if response.text:
            # Directly parse the JSON response
            clean_response = response.text.replace('\n', ' ').split(' | ')

            return clean_response
        else:
            raise Exception("The response from the model was empty.")

    except Exception as e:
        print(f"Error occurred: {e}")
        return "Could not generate a suggestion at this time."
