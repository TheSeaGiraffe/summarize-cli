from pathlib import Path

import pytest
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.prompts import ChatPromptTemplate


# @pytest.fixture(scope="class")
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
def pdf_file_word_counts():
    pdf_file_word_counts = {
        "1810.04805v2": 9777,
        "2106.09685v2": 12531,
        "2310.06825v1": 3812,
        "2412.19437v2": 24908,
        "NIPS-2017-attention-is-all-you-need-Paper": 5413,
    }

    return pdf_file_word_counts


@pytest.fixture(scope="function")
def fake_summarization_chain():
    prompt_text = (
        "This is just a dummy prompt but summarize this text anyway: {context}"
    )
    prompt = ChatPromptTemplate.from_template(prompt_text)

    responses = ["This is a fake summary"]
    llm = FakeListChatModel(responses=responses)

    return create_stuff_documents_chain(llm, prompt)
