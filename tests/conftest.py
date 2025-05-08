from pathlib import Path

import pytest
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.prompts import ChatPromptTemplate


@pytest.fixture()
def debug_env_var(monkeypatch):
    monkeypatch.setenv("SUMMARIZE_CLI_TEST", "1")
    yield
    monkeypatch.delenv("SUMMARIZE_CLI_TEST")


@pytest.fixture(scope="function")
def pdf_files():
    test_root = Path(__file__).parent
    pdf_dir = test_root / "pdfs"
    return [f for f in pdf_dir.iterdir()]


@pytest.fixture(scope="function")
def fake_summarization_chain():
    prompt_text = (
        "This is just a dummy prompt but summarize this text anyway: {context}"
    )
    prompt = ChatPromptTemplate.from_template(prompt_text)

    responses = ["This is a fake summary"]
    llm = FakeListChatModel(responses=responses)

    return create_stuff_documents_chain(llm, prompt)
