import os
import google.generativeai as genai
from src.utils import config

class GeminiHandler:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

        self.model_name = config.LLM_MODEL_NAME_GEMINI
        print(f"Initializing Google Gemini client with model '{self.model_name}'...")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            self.model_name,
            system_instruction="You are a helpful AI assistant for a sales team."
        )
        print("Google Gemini client initialized.")

    def get_answer(self, prompt: str) -> str:
        print("Sending request to Google Gemini API...")
        try:
            response = self.model.generate_content(prompt)
            print("Response received from Google Gemini API.")
            return response.text.strip()
        except Exception as e:
            print(f"An error occurred while calling the Google Gemini API: {e}")
            return "Error: Could not get a response from the AI model."