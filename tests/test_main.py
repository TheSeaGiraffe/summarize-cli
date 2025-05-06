from pathlib import Path
from typing import Any

import pytest
from langchain_core.runnables import Runnable

from summarize_cli.cli import check_api_key_var
from summarize_cli.summarize import summarize_pdfs


class TestMainGeneral:
    """Tests general app functionality."""

    # Wondering if I shouldn't just call the main command here. Will leave it like this
    # for now.
    def test_no_api_key(self, capsys, debug_env_var):
        """Tests that the check for an API key works properly."""

        with pytest.raises(SystemExit) as cap_err:
            # check_api_key_var(debug=True)
            check_api_key_var()
        output = capsys.readouterr().out
        assert cap_err.value.code == 1
        assert "Missing API key" in output

    # Do I even need to bother with types here?
    def fake_summarize_pdfs(
        self,
        chain: Runnable[dict[str, Any], Any],
        pdfs: list[Path],
        output_path: Path,
        suffix: str = "summary",
    ) -> None:
        summarize_pdfs(chain, pdfs, output_path, suffix)

    @pytest.mark.parametrize(
        "output_dir", [Path("."), Path("test_d1"), Path("test_d3/one/two")]
    )
    def test_output_dir(
        self, tmp_path, output_dir, pdf_files, fake_summarization_chain, debug_env_var
    ):
        """Tests that the output dir gets created"""

        output_path = tmp_path / output_dir
        self.fake_summarize_pdfs(fake_summarization_chain, pdf_files, output_path)

        assert output_path.exists()

    @pytest.mark.parametrize("num_files", [1, 3, 5])
    def test_output_files(
        self, num_files, tmp_path, pdf_files, fake_summarization_chain, debug_env_var
    ):
        """Tests that the number of summary files matches the number of pdf files"""

        pdf_slice = pdf_files[num_files] if num_files == 0 else pdf_files[:num_files]
        self.fake_summarize_pdfs(fake_summarization_chain, pdf_slice, tmp_path)

        assert len(list(tmp_path.iterdir())) == num_files

    @pytest.mark.parametrize(
        "suffix", ["poodonkis", "concise-summary", "fake_summary", "funky@summary", ""]
    )
    def test_output_file_suffix(
        self, suffix, tmp_path, pdf_files, fake_summarization_chain, debug_env_var
    ):
        """Test that the output files have the specified suffixes"""

        self.fake_summarize_pdfs(fake_summarization_chain, pdf_files, tmp_path, suffix)

        for i, f in enumerate(sorted(tmp_path.iterdir())):
            if suffix == "":
                pdf_name = pdf_files[i].stem
                assert pdf_name == f.stem
            else:
                assert suffix in f.stem


class TestMainSummarization:
    """Tests that the summarization functionality works as expected."""

    def test_always_pass(self):
        assert True
