from pathlib import Path
from typing import Any

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from pymupdf import TOOLS
from tqdm import tqdm


def get_pdf_text(
    pdf_file: str | Path,
    single_mode: bool = False,
    page_delim: str | None = None,
) -> list[Document]:
    if single_mode:
        default_delim = "\n-----\n\n"
        if page_delim is None:
            page_delim = default_delim

        loader = PyMuPDF4LLMLoader(pdf_file, mode="single", pages_delimiter=page_delim)
    else:
        loader = PyMuPDF4LLMLoader(pdf_file)
    doc = loader.load()

    return doc


def gen_stuff_summary_chain_with_prompt(
    model_name: str, model_provider: str, prompt_text: str | None = None
) -> Runnable[dict[str, Any], Any]:
    llm = init_chat_model(model_name, model_provider=model_provider)
    if prompt_text is None:
        prompt_text = "Write a concise summary of the following: {context}"
    prompt = ChatPromptTemplate.from_template(prompt_text)
    return create_stuff_documents_chain(llm, prompt)


def summarize_pdfs(
    stuff_documents_chain: Runnable[dict[str, Any], Any],
    pdf_files: list[Path],
    output_dir: Path,
    output_file_suffix: str,
) -> None:
    # Loop through the files and create summaries for them
    # Will need to add either file extension or MIME type checks for each file as an
    # additional layer of validation
    TOOLS.mupdf_display_errors(False)  # Ignore errors from PyMuPDF

    # Create the output directory if the user passed one in and it doesn't exist.
    # Otherwise, use the current directory.
    # This should work even if the path is the current directory
    # Not sure if this should even be here. Will keep it here for now to simplify
    # testing.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate summaries
    for pdf_file in tqdm(pdf_files):
        # Generate summary
        doc = get_pdf_text(pdf_file, single_mode=True, page_delim="")
        summary_text = stuff_documents_chain.invoke({"context": doc})

        # Write summary to file
        with open(output_dir / f"{pdf_file.stem}-{output_file_suffix}.txt", "w") as out:
            _ = out.write(summary_text)
