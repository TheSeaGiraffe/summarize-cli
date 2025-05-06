import os
import sys
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv

from summarize_cli.summarize import (
    SUMMARY_PROMPTS,
    gen_stuff_summary_chain_with_prompt,
    summarize_pdfs,
)


@click.command()
@click.help_option("-h", "--help")
@click.argument(
    "files",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--summary-type",
    "-s",
    type=click.Choice(["concise", "bullet_point", "detailed"], case_sensitive=False),
    default="concise",
    help="The type of summary that will be generated",
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
def main(files: list[Path], summary_type: str, output_dir: Path, suffix: str) -> None:
    """
    Takes in one or more journal article FILES as input and produces a summary for each
    one using an LLM. Currently, only the GPT-4o mini model is supported.
    """
    # Check for API key. Don't proceed if one isn't found
    check_api_key_var()

    # Create the stuff documents chain
    # Hardcode these for now
    llm_model = "gpt-4o-mini"
    model_provider = "openai"
    prompt_text = SUMMARY_PROMPTS[summary_type.lower()]
    chain = gen_stuff_summary_chain_with_prompt(llm_model, model_provider, prompt_text)

    # Loop through the files and create summaries for them
    summarize_pdfs(chain, files, output_dir, suffix)


def check_api_key_var() -> None:
    # Load .env file if one is present
    debug = os.getenv("SUMMARIZE_CLI_DEBUG", "0")
    if debug != "1":
        env_file = find_dotenv()
        if env_file:
            _ = load_dotenv(env_file)

    # Check for API key. Don't proceed if one isn't found
    if "OPENAI_API_KEY" not in os.environ:
        click.echo(
            "Missing API key. Make sure the appropriate environment variable "
            "is set or that a .env file containing the variable is present "
            "in the current directory."
        )
        sys.exit(1)
