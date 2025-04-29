import os
import sys
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pymupdf import TOOLS
from tqdm import tqdm

from summarize_cli.pdftext import get_pdf_text


@click.command()
@click.help_option("-h", "--help")
@click.argument(
    "files",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--output-dir",
    "-d",
    default=".",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Path to the directory that will be used to store the output. If none specified will output all summaries to the current directory.",
)
@click.option(
    "--suffix",
    type=str,
    default="summary",
    show_default=True,
    help="Suffix to use for the output files.",
)
def main(files: list[Path], output_dir: Path, suffix: str) -> None:
    """
    Takes in one or more journal article FILES as input and produces a summary for each
    one using an LLM. Currently, only the GPT-4o mini model is supported.
    """
    # Load .env file if one is present
    env_file = find_dotenv()
    if env_file:
        _ = load_dotenv(env_file)

    # Check for API key. Don't proceed if one isn't found
    if "OPENAI_API_KEY" not in os.environ:
        click.echo(
            "Missing API key. Make sure the appropriate environment variable "
            "is set or than a .env file containing the variable is present "
            "in the current directory."
        )
        sys.exit(1)

    # Create the output directory if the user passed one in and it doesn't exist.
    # Otherwise, use the current directory.
    # This should work even if the path is the current directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the model
    llm = init_chat_model("gpt-4o-mini", model_provider="openai")
    prompt_text = "Write a concise summary of the following: {context}"
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Loop through the files and create summaries for them
    # Will need to add either file extension or MIME type checks for each file as an
    # additional layer of validation
    TOOLS.mupdf_display_errors(False)  # Ignore errors from PyMuPDF
    for file in tqdm(files):
        # Generate summary
        doc = get_pdf_text(file, single_mode=True, page_delim="")
        chain = create_stuff_documents_chain(llm, prompt)
        summary_text = chain.invoke({"context": doc})

        # Write summary to file
        with open(output_dir / f"{file.stem}-{suffix}.txt", "w") as out:
            _ = out.write(summary_text)
