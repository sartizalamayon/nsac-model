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


