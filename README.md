# Sales Copilot: AI Chatbot for Sales Call Analysis

Sales Copilot is a command-line chatbot designed to help sales professionals understand and summarize their past sales calls. The tool ingests call transcripts, stores them in a searchable knowledge base, and allows users to ask natural language questions, generate summaries, and retrieve specific information with source attribution.

## Features

* **Interactive Chat Shell**: Run the application once to enter an interactive session where you can run multiple commands without reloading the AI models.

* **Data Ingestion**: Ingest single or multiple call transcript files from a directory. The data is parsed, chunked, and stored as vector embeddings for fast semantic search.

* **Retrieval-Augmented Generation (RAG)**: Ask questions in natural language (e.g., "What were the customer's main objections?"). The chatbot retrieves the most relevant segments from the call transcripts and uses them to generate a precise, source-cited answer.

* **Full Call Summarization**: Generate a comprehensive, structured summary of an entire call, detailing the main purpose, key discussion points, customer sentiment, and action items.

* **Flexible AI Backend**: Easily switch between different powerful Large Language Models (LLMs) like OpenAI's GPT series or Google's Gemini series by changing a single configuration setting.

## Architecture

This project is built with a modular and scalable architecture, following SOLID principles.

* **Storage**: Vector embeddings are stored in a [FAISS](https://faiss.ai/) index for efficient similarity search, with corresponding metadata stored in a separate file.

* **Retrieval**: The core logic uses a Retrieval-Augmented Generation (RAG) pattern. Text embeddings are generated using the `sentence-transformers` library.

* **Modularity**: The application is broken down into distinct, single-responsibility components for ingestion, retrieval, LLM interaction, and prompt engineering.

* **LLM Agnostic**: A factory pattern is used to initialize the LLM handler, allowing seamless switching between providers (`OpenAI`, `Gemini`) via a configuration setting.

## Setup and Installation

Follow these steps to set up and run the project locally.

### 1. Prerequisites

* Python 3.8 or higher

* `pip` for package management

### 2. Clone the Repository

```
git clone <your-repository-url>
cd sales-copilot

```

### 3. Set Up a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

```
# Create the virtual environment
python -m venv venv

# Activate it
# On Windows:
# venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

```

### 4. Install Dependencies

Install all the required Python libraries from the `requirements.txt` file.

```
pip install -r requirements.txt

```

### 5. Configure Environment Variables

The application uses a `.env` file to manage API keys and model configurations.

a. **Create** the `.env` **file:** Copy the example file to create your local configuration.

```
cp .env.example .env

```

b. **Edit the `.env` file:** Open the newly created `.env` file and add your API keys and desired configuration.

```
# LLM_PROVIDER can be 'openai' or 'gemini'
LLM_PROVIDER="gemini"

# --- Model Names ---
LLM_MODEL_NAME_OPENAI="gpt-3.5-turbo"
LLM_MODEL_NAME_GEMINI="gemini-1.5-flash-latest"

# --- API Keys ---
# Add your secret API keys here
OPENAI_API_KEY="sk-..."
GOOGLE_API_KEY="AIzaSy..."

```

* **`LLM_PROVIDER`**: This is the most important setting. Set it to `openai` to use OpenAI's models or `gemini` to use Google's models.

* Make sure to add the correct API key for the provider you have selected.

### 6. Add Transcript Data

Place your call transcript `.txt` files inside the `data/` directory. The project includes sample files to get you started.

## How to Use

The application is designed to be run as an interactive shell by default.

### 1. Start the Interactive Session

To start the chatbot, simply run `cli.py` from the root of the project.

```
python cli.py

```

This will load the configured AI models and drop you into the interactive `(copilot)>` prompt.

### 2. Interactive Commands

Once in the shell, you can use the following commands:

* **`ingest <path>`**: Ingest new transcripts. The path can be to a single `.txt` file or a directory.

  ```
  (copilot)> ingest data/
  
  ```

* **`list`**: List all the call IDs that have been ingested.

  ```
  (copilot)> list
  
  ```

* **`ask <question>`**: Ask a question about the content of the calls.

  ```
  (copilot)> ask What was the final agreed upon TCV?
  
  ```

* **`summarise <call_id>`**: Generate a full, structured summary of a specific call.

  ```
  (copilot)> summarise 4_negotiation_call.txt
  
  ```

* `exit` or **`quit`**: End the interactive session.

### 3. Non-Interactive (Scripting) Use

You can also run commands as single, one-off tasks directly from your terminal. This is useful for scripting and automation.

```
# Ingest data and then exit
python cli.py ingest data/

# Ask a single question and then exit
python cli.py ask "What were the main security concerns?"

```

## Running Tests

The project includes a suite of unit and integration tests to ensure the core components are working correctly. To run the tests, use `pytest`.

```
pytest

```

## Assumptions

* **Transcript Format**: The parser is designed to handle transcripts with lines in the format `[HH:MM] Speaker (Role): Text...`. It correctly handles multi-line text from a single speaker.

* **API Keys**: The application assumes you have valid, active API keys for the selected LLM provider (OpenAI or Google AI).

* **File Structure**: The application expects the directory structure as laid out in the repository (e.g., `src/`, `data/`, `prompts/`).

## Project Structure

```
sales-copilot/
├── data/                 # Raw transcript files
├── models/               # (Optional) For local GGUF models
├── prompts/              # Prompt templates (.txt files)
├── src/
│   ├── ingestion/
│   │   ├── parser.py
│   │   ├── chunker.py
│   │   └── ingestion_pipeline.py
│   ├── retrieval/
│   │   └── retrieval_pipeline.py
│   ├── storage/
│   │   └── vector_store.py
│   ├── llm/
│   │   ├── llm_handler.py      # Factory for LLM providers
│   │   ├── openai_handler.py
│   │   ├── gemini_handler.py
│   │   └── prompt_builder.py
│   └── utils/
│       └── config.py
├── tests/                # Unit and integration tests
├── .env                  # Local environment variables (DO NOT COMMIT)
├── .env.example          # Template for environment variables
├── .gitignore
├── cli.py                # Main application entry point
├── requirements.txt
└── README.md
```
