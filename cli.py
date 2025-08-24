import click
import os
from tqdm import tqdm
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.ingestion.ingestion_pipeline import IngestionPipeline
from src.llm.llm_handler import LLMHandler
from src.llm.prompt_builder import PromptBuilder
from src.storage.vector_store import VectorStore
from src.utils import config


class SalesCopilot:
    """A controller class to orchestrate the chatbot's functionalities."""

    def __init__(self):
        click.echo("Initializing Sales Copilot... (this may take a moment)")

        vector_store = VectorStore(
            index_path=config.VECTOR_STORE_INDEX_PATH,
            metadata_path=config.VECTOR_STORE_METADATA_PATH,
            embedding_dim=config.EMBEDDING_DIMENSION
        )
        self.ingestor = IngestionPipeline(vector_store)
        self.retriever = RetrievalPipeline(vector_store)
        self.llm = LLMHandler()
        self.prompt_builder = PromptBuilder()
        click.echo("Initialization complete. Welcome to Sales Copilot!")

    def ingest(self, path: str):
        """Handles the ingestion of files or directories."""
        files_to_ingest = []
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith(".txt"):
                        files_to_ingest.append(os.path.join(root, file))
        elif os.path.isfile(path) and path.endswith(".txt"):
            files_to_ingest.append(path)

        if not files_to_ingest:
            click.echo(f"Error: No .txt files found at path: {path}", err=True)
            return

        click.echo(f"Found {len(files_to_ingest)} files to ingest.")
        file_iterator = tqdm(files_to_ingest, desc="Preparing Ingestion", colour='green')
        for file_path in file_iterator:
            call_id = os.path.basename(file_path)
            file_iterator.set_description(f"Ingesting {call_id}")
            self.ingestor.run(file_path, call_id)
        # Save the store once after all files are processed
        self.ingestor.store.save()
        click.echo(f"\nIngestion complete.")

    def summarise(self, call_id: str):
        """
        Generates a direct summary for a full call transcript in a single API call.
        """
        click.echo(f"Generating summary for '{call_id}'...")

        # 1. Retrieve the full transcript text from the vector store.
        transcript = self.retriever.store.get_full_transcript(call_id)
        if not transcript:
            click.echo(f"Error: Call ID '{call_id}' not found.", err=True)
            return

        # 2. Build the simple, direct prompt with the full transcript.
        prompt = self.prompt_builder.build_summary_prompt(transcript)

        # 3. Call the LLM once to get the final summary.
        final_summary = self.llm.get_answer(prompt)

        # 4. Print the result.
        click.echo("\n" + "=" * 20 + f" SUMMARY: {call_id} " + "=" * 20)
        click.echo(final_summary)
        click.echo("=" * (42 + len(call_id)))

    def ask(self, query: str):
        """Handles the Q&A (RAG) functionality."""
        click.echo(f"Searching for an answer to: '{query}'...")
        context_chunks = self.retriever.retrieve_relevant_docs(query)

        if not context_chunks:
            click.echo("\nCould not find any relevant information to answer your question.")
            return

        prompt = self.prompt_builder.build_qa_prompt(query, context_chunks)
        answer = self.llm.get_answer(prompt)

        click.echo("\n" + "=" * 20 + " ANSWER " + "=" * 20)
        click.echo(answer)
        click.echo("\n" + "=" * 20 + " SOURCES " + "=" * 20)
        for i, chunk in enumerate(context_chunks):
            click.echo(f"\n--- Source [{i + 1}] ---")
            click.echo(f"Call ID: {chunk['call_id']}")
            click.echo(f"Timestamp: {chunk.get('timestamp', 'N/A')}")
            click.echo(f"Speaker: {chunk['speaker']}")
            click.echo(f"Text: \"...{chunk['text'][:250].strip()}...\"")
        click.echo("=" * 49)


# --- Click Command Definitions ---

@click.group(invoke_without_command=True, context_settings=dict(help_option_names=['-h', '--help']))
@click.pass_context
def cli(ctx):
    """
    Sales Copilot CLI.

    Run without a command to enter interactive mode.
    """
    # 2. Check if a subcommand was invoked. If not, start the interactive shell.
    if ctx.invoked_subcommand is None:
        copilot = SalesCopilot()
        click.echo("Interactive session started. Type 'exit' or 'quit' to end.")

        while True:
            user_input = click.prompt("(copilot)")
            if user_input.lower() in ['exit', 'quit']:
                click.echo("Exiting session.")
                break

            parts = user_input.strip().split(maxsplit=1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            try:
                if command == 'ask':
                    if not args: click.echo("Error: 'ask' requires a question.", err=True); continue
                    copilot.ask(args)
                elif command == 'summarise':
                    if not args: click.echo("Error: 'summarise' requires a call_id.", err=True); continue
                    copilot.summarise(args)
                elif command == 'list':
                    call_ids = copilot.retriever.store.get_all_call_ids()
                    if not call_ids:
                        click.echo("No calls have been ingested yet.")
                    else:
                        click.echo("Available call IDs:")
                        for cid in sorted(list(call_ids)): click.echo(f"- {cid}")
                elif command == 'ingest':
                    if not args:
                        click.echo("Error: 'ingest' requires a file or directory path.", err=True)
                        continue
                    path = args
                    if not os.path.exists(path):
                        click.echo(f"Error: Path does not exist: '{path}'", err=True)
                        continue
                    copilot.ingest(path)
                else:
                    click.echo(f"Unknown command: '{command}'. Available: ingest, ask, summarise, list, exit")
            except Exception as e:
                click.echo(f"An error occurred: {e}", err=True)


# The subcommands still exist for single, non-interactive use
@cli.command()
@click.argument('path', type=click.Path(exists=True))
def ingest(path):
    """Ingest a transcript file or a directory of .txt files."""
    copilot = SalesCopilot()
    copilot.ingest(path)


@cli.command(name="list")
def list_calls():
    """List all ingested call IDs."""
    # PROVIDE the embedding_dim here as well
    store = VectorStore(
        index_path=config.VECTOR_STORE_INDEX_PATH,
        metadata_path=config.VECTOR_STORE_METADATA_PATH,
        embedding_dim=config.EMBEDDING_DIMENSION
    )
    # This command logic can be simplified as it doesn't need a full pipeline
    call_ids = store.get_all_call_ids()
    if not call_ids:
        click.echo("No calls have been ingested yet.")
        return
    click.echo("Available call IDs:")
    for call_id in sorted(list(call_ids)):
        click.echo(f"- {call_id}")


@cli.command()
@click.argument('call_id')
def summarise(call_id):
    """Summarise a specific call by its ID."""
    copilot = SalesCopilot()
    copilot.summarise(call_id)


@cli.command()
@click.argument('question')
def ask(question):
    """Ask a question about the content of the sales calls."""
    copilot = SalesCopilot()
    copilot.ask(question)


# The standalone `chat` command is no longer needed.

if __name__ == '__main__':
    cli()