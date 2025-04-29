# `summarize-cli`

This is a fairly simple CLI app designed to summarize journal articles using an LLM. So
far `summarize-cli` only supports OpenAI models but support for other models may be
included in the future. `summarize-cli` assumes that any journal articles passed to it are
in PDF format. Support for other formats will be included soon.

## Installation

Still working on making this into a proper package. For now, clone the repo and make sure
you have `uv` installed on your system. You can then run `summarize-cli` using the
`uv run` command:

```bash
uv run summarize-cli -h
```

## Basic usage

You can summarize a bunch of journal articles by passing them to the CLI:

```bash
uv run summarize-cli article01.pdf article02.pdf article03.pdf
```

This command will generate summaries for each article and then place them in the current
directory by default. You can also specify an output directory using the `--output-dir`
flag. `summarize-cli` will create the directory along with the necessary parent
directories if it doesn't exist. Additionally, you can specify a suffix to be attached to
the names of each summary file using the `--suffix` flag. The default is `summary`.
