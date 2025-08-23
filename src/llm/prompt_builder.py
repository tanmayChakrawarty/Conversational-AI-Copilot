from typing import List, Dict
from src.utils import config

class PromptBuilder:
    def __init__(self, template_dir: str = config.PROMPT_TEMPLATE_DIR):
        try:
            with open(f"{template_dir}/{config.QA_PROMPT_TEMPLATE}", "r", encoding='utf-8') as f:
                self.qa_template = f.read()
            with open(f"{template_dir}/summary.txt", "r", encoding='utf-8') as f:
                self.summary_template = f.read()
        except FileNotFoundError as e:
            raise FileNotFoundError(f"A prompt template was not found: {e}.")

    # --- build_qa_prompt and build_focused_summary_prompt remain the same ---

    def build_qa_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        context_str = ""
        for i, chunk in enumerate(context_chunks):
            source_info = f"Call ID: {chunk['call_id']}, Speaker: {chunk['speaker']}"
            context_str += f"Source [{i+1}] ({source_info}):\n{chunk['text']}\n\n"
        return self.qa_template.format(question=query, context=context_str.strip())

    # NEW: A single, simple method for building the summary prompt
    def build_summary_prompt(self, transcript: str) -> str:
        """Constructs a prompt for a direct, single-shot summarization."""
        return self.summary_template.format(transcript=transcript)