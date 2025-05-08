import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from summarize_cli.cli import check_api_key_var, main
from summarize_cli.summarize import (
    get_pdf_text,
    summarize_pdfs_async,
)


class TestMainAsyncGeneral:
    """Tests general app functionality."""

    # Wondering if I shouldn't just call the main command here. Will leave it like this
    # for now.
    def test_no_api_key(self, capsys, debug_env_var):
        """Tests that the check for an API key works properly."""

        with pytest.raises(SystemExit) as cap_err:
            check_api_key_var()
        output = capsys.readouterr().out
        assert cap_err.value.code == 1
        assert "Missing API key" in output

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "output_dir", [Path("."), Path("test_d1"), Path("test_d3/one/two")]
    )
    async def test_output_dir(
        self, tmp_path, output_dir, pdf_files, fake_summarization_chain, debug_env_var
    ):
        """Tests that the output dir gets created"""

        output_path = tmp_path / output_dir
        suffix = "summary"
        await summarize_pdfs_async(
            fake_summarization_chain, pdf_files, output_path, suffix
        )

        assert output_path.exists()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("num_files", [1, 3, 5])
    async def test_output_files(
        self, num_files, tmp_path, pdf_files, fake_summarization_chain, debug_env_var
    ):
        """Tests that the number of summary files matches the number of pdf files"""

        pdf_slice = pdf_files[num_files] if num_files == 0 else pdf_files[:num_files]
        suffix = "summary"
        await summarize_pdfs_async(
            fake_summarization_chain, pdf_slice, tmp_path, suffix
        )

        assert len(list(tmp_path.iterdir())) == num_files

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "suffix", ["poodonkis", "concise-summary", "fake_summary", "funky@summary", ""]
    )
    async def test_output_file_suffix(
        self, suffix, tmp_path, pdf_files, fake_summarization_chain, debug_env_var
    ):
        """Test that the output files have the specified suffixes"""

        await summarize_pdfs_async(
            fake_summarization_chain, pdf_files, tmp_path, suffix
        )

        for i, f in enumerate(sorted(tmp_path.iterdir())):
            pdf_name = pdf_files[i].stem
            if suffix == "":
                assert pdf_name == f.stem
            else:
                assert f"{pdf_name}-{suffix}" == f.stem


class TestMainAsyncSummarization:
    """Tests that the summarization functionality works as expected."""

    def get_word_count(self, doc: str) -> int:
        return len(re.split(r"\s+", doc))

    # Need to figure out how to use async functions here in conjunction with the runner
    # call to the main function.
    def test_generate_summaries_concise(self, tmp_path, pdf_files):
        """Test summarization using the default 'concise' summary type"""
        runner = CliRunner()
        invoke_opts = ["--asynchronous", "--output-dir", str(tmp_path)]
        # We test only the first 3 pdf files to save time
        invoke_opts += [str(f) for f in pdf_files[:3]]
        result = runner.invoke(main, invoke_opts)

        assert result.exit_code == 0

        for summary_file, pdf_file in zip(sorted(tmp_path.iterdir()), pdf_files[:3]):
            # Get pdf text
            doc = get_pdf_text(pdf_file, single_mode=True, page_delim="")
            pdf_text = doc[0].page_content

            # Get summary text
            with open(summary_file) as summary:
                summary_text = summary.read()

            assert self.get_word_count(summary_text) < self.get_word_count(pdf_text)
