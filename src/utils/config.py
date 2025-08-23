import os
from dotenv import load_dotenv

load_dotenv()

# --- Embedding Model Configuration ---
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = os.getenv("EMBEDDING_DIMENSION", 384)

# --- Model Configuration ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini") # Default to gemini
LLM_MODEL_NAME_OPENAI = os.getenv("LLM_MODEL_NAME_OPENAI", "gpt-3.5-turbo")
LLM_MODEL_NAME_GEMINI = os.getenv("LLM_MODEL_NAME_GEMINI", "gemini-1.5-flash-latest")

# --- Vector Storage Configuration ---
VECTOR_STORE_INDEX_PATH = "vector_data/vector_store.faiss"
VECTOR_STORE_METADATA_PATH = "vector_data/metadata.pkl"

# --- Prompt Configuration ---
PROMPT_TEMPLATE_DIR = "prompts/"
QA_PROMPT_TEMPLATE = "qa.txt"
SUMMARY_PROMPT_TEMPLATE = "summary.txt"