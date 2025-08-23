import os
from openai import OpenAI
from src.utils import config

class OpenAIHandler:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")

        self.model_name = config.LLM_MODEL_NAME_OPENAI
        print(f"Initializing OpenAI client with model '{self.model_name}'...")
        self.client = OpenAI(api_key=api_key)
        print("OpenAI client initialized.")

    def get_answer(self, prompt: str) -> str:
        print("Sending request to OpenAI API...")
        try:
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant for a sales team."},
                {"role": "user", "content": prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
            )
            print("Response received from OpenAI API.")
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"An error occurred while calling the OpenAI API: {e}")
            return "Error: Could not get a response from the AI model."