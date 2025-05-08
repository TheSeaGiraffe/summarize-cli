import asyncio
import os
from pathlib import Path
from typing import Any

import aiofiles
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from pymupdf import TOOLS
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

__all__ = ["get_pdf_text", "gen_stuff_summary_chain_with_prompt", "summarize_pdfs"]


SUMMARY_PROMPTS = {
    "concise": "Write a concise summary of the following: {context}",
    "bullet_point": "Write a concise summary of the following and have the output be in the form of bullet points: {context}",
    "detailed": "Write a detailed summary of the following: {context}",
}


def get_pdf_text(
    pdf_file: str | Path,
    single_mode: bool = False,
    page_delim: str | None = None,
) -> list[Document]:
    """Extract text from a pdf file.

    Text is extracted using the 3rd party tool PyMuPDF4LLMLoader.

    Parameters
    ----------
    pdf_file: str | Path
        Path to pdf file
    single_mode: bool
        Whether the text should be extracted as a single document or split up into
        multiple documents. Default: False
    page_delim: str | None
        The delimiter that will be used to separate pages if "single_mode" is specified.
        If None, then a simple bar of 5 hyphens is used. Default: None

    Returns
    -------
    list[Document]
        A list of LangChain Document objects.
    """
    test = os.getenv("SUMMARIZE_CLI_TEST", "0")
    doc: list[Document]
    if test == "1":
        doc = [
            Document(
                page_content="This is a fake document", metadata={"title": "fake_doc"}
            )
        ]
    else:
        if single_mode:
            default_delim = "\n-----\n\n"
            if page_delim is None:
                page_delim = default_delim

            loader = PyMuPDF4LLMLoader(
                pdf_file, mode="single", pages_delimiter=page_delim
            )
        else:
            loader = PyMuPDF4LLMLoader(pdf_file)
        doc = loader.load()

    return doc


async def get_pdf_text_async(
    pdf_file: str | Path,
    single_mode: bool = False,
    page_delim: str | None = None,
) -> list[Document]:
    """Extract text from a pdf file asynchronously.

    Text is extracted using the 3rd party tool PyMuPDF4LLMLoader.

    Parameters
    ----------
    pdf_file: str | Path
        Path to pdf file
    single_mode: bool
        Whether the text should be extracted as a single document or split up into
        multiple documents. Default: False
    page_delim: str | None
        The delimiter that will be used to separate pages if "single_mode" is specified.
        If None, then a simple bar of 5 hyphens is used. Default: None

    Returns
    -------
    list[Document]
        A list of LangChain Document objects.
    """
    test = os.getenv("SUMMARIZE_CLI_TEST", "0")
    doc: list[Document]
    if test == "1":
        await asyncio.sleep(1)
        doc = [
            Document(
                page_content="This is a fake document", metadata={"title": "fake_doc"}
            )
        ]
    else:
        if single_mode:
            default_delim = "\n-----\n\n"
            if page_delim is None:
                page_delim = default_delim

            loader = PyMuPDF4LLMLoader(
                pdf_file, mode="single", pages_delimiter=page_delim
            )
        else:
            loader = PyMuPDF4LLMLoader(pdf_file)

        doc = await loader.aload()

    return doc


def gen_stuff_summary_chain_with_prompt(
    model_name: str, model_provider: str, prompt_text: str
) -> Runnable[dict[str, Any], Any]:
    """Create a LangChain text summarization chain with the provided model and prompt

    Parameters
    ----------
    model_name: str
        The name of the model that will be used
    model_provider: str
        The name of the organization/company behind the model. Basically, the "class" of
        model
    prompt_text: str
        The prompt that will be used to generate the summary

    Returns
    -------
    Runnable[dict[str, Any], Any]
        A LangChain runnable object.
    """
    llm = init_chat_model(model_name, model_provider=model_provider)
    prompt = ChatPromptTemplate.from_template(prompt_text)
    return create_stuff_documents_chain(llm, prompt)


def summarize_pdfs(
    stuff_documents_chain: Runnable[dict[str, Any], Any],
    pdf_files: list[Path],
    output_dir: Path,
    output_file_suffix: str,
) -> None:
    """Summarize a batch of pdf files

    Summaries are generated using the provided summarization chain and are written to a
    text file in the specified `output_dir`. The names of the summary files should have
    the specified `output_file_suffix` attached to them.

    Parameters
    ----------
    stuff_documents_chain: Runnable[dict[str, Any], Any]
        The text summarization chain that will be used to generate the summaries
    pdf_files: list[Path]
        A list of pdf files that will be used to generate summaries
    output_dir: Path
        The directory in which the generated summaries will be stored
    output_file_suffix: str
        The suffix that will be added to the names of all of the summary files
    """
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
        output_filename = f"{pdf_file.stem}"
        if output_file_suffix:
            output_filename += f"-{output_file_suffix}"
        output_filename += ".txt"
        with open(output_dir / output_filename, "w") as out:
            _ = out.write(summary_text)


async def summarize_pdfs_async(
    stuff_documents_chain: Runnable[dict[str, Any], Any],
    pdf_files: list[Path],
    output_dir: Path,
    output_file_suffix: str,
) -> None:
    """Summarize a batch of pdf files asynchronously

    Summaries are generated using the provided summarization chain and are written to a
    text file in the specified `output_dir`. The names of the summary files should have
    the specified `output_file_suffix` attached to them.

    Parameters
    ----------
    stuff_documents_chain: Runnable[dict[str, Any], Any]
        The text summarization chain that will be used to generate the summaries
    pdf_files: list[Path]
        A list of pdf files that will be used to generate summaries
    output_dir: Path
        The directory in which the generated summaries will be stored
    output_file_suffix: str
        The suffix that will be added to the names of all of the summary files
    """
    TOOLS.mupdf_display_errors(False)  # Ignore errors from PyMuPDF

    # Get pdfs
    print("Getting pdf texts...")
    get_pdf_tasks = []
    for pdf_file in pdf_files:
        get_pdf_tasks.append(
            get_pdf_text_async(pdf_file, single_mode=True, page_delim="")
        )
    docs = await atqdm.gather(*get_pdf_tasks)

    # Generate summaries
    print("\nGenerating summaries...")
    summary_tasks = []
    for doc in docs:
        summary_tasks.append(stuff_documents_chain.ainvoke({"context": doc}))
    summaries = await atqdm.gather(*summary_tasks)

    # Write summaries to file
    print("\nWriting summaries to individual files...")
    output_dir.mkdir(parents=True, exist_ok=True)
    num_files = len(pdf_files)
    for pdf_file, summary in tqdm(zip(pdf_files, summaries), total=num_files):
        output_filename = f"{pdf_file.stem}"
        if output_file_suffix:
            output_filename += f"-{output_file_suffix}"
        output_filename += ".txt"

        async with aiofiles.open(output_dir / output_filename, "w") as out:
            await out.write(summary)

    print("Done.")
