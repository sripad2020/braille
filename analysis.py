import os
import google.generativeai as genai
from dotenv import load_dotenv # Only if you are using .env file
from PIL import Image # Import the Pillow library for image handling

# Load environment variables from .env file (if you are using .env file)
load_dotenv()

# Configure the API key
# The library will automatically look for GOOGLE_API_KEY in environment variables

genai.configure(api_key="AIzaSyC9lsET5jCJJOZmoPQ8k8TeMqeYvTvhIfk") # Replace with your actual key or use environment variables

model = genai.GenerativeModel('gemini-1.5-flash')

# --- Image Handling Example ---
print("--- Image Content Analysis ---")
image_path = 'braille_abcde.png'

# Check if the image file exists
if not os.path.exists(image_path):
    print(f"Error: Image file not found at '{image_path}'")
else:
    try:
        img = Image.open(image_path)
        image_prompt_parts = [
            img,
            "explain me about this braille image in detail give me results of its complete analysis"
        ]

        image_response = model.generate_content(image_prompt_parts)
        print(f"User: (Image: {image_path}) What do you see in this image? Describe it in detail.")
        print(f"Gemini: {image_response.text}")
        print("-" * 30)
        image_prompt_parts_2 = [
            img,
            "Can you identify any text or symbols in this image?"
        ]
        image_response_2 = model.generate_content(image_prompt_parts_2)
        print(f"User: (Image: {image_path}) Can you identify any text or symbols in this image?")
        print(f"Gemini: {image_response_2.text}")
        print("-" * 30)


    except Exception as e:
        print(f"An error occurred during image processing: {e}")
        print("Please ensure the image path is correct and Pillow (PIL) is installed (`pip install Pillow`).")

# --- Original Text Generation Examples (kept for completeness) ---
