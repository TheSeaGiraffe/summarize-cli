from pathlib import Path

from langchain_core.documents import Document
from langchain_pymupdf4llm import PyMuPDF4LLMLoader


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
