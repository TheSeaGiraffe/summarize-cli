# `summarize-cli`

This is a fairly simple CLI app designed to summarize journal articles using an LLM. This
is something that I made mostly for myself for fun as I often find myself reading articles
and wishing that I had a quick and easy way to summarize them. So far `summarize-cli` only
supports OpenAI's GPT-4o mini model but support for other models may be included in the
future. `summarize-cli` assumes that any journal articles passed to it are in PDF format.

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
flag.

```bash
uv run summarize-cli --output-dir article_summaries article01.pdf article02.pdf
```

`summarize-cli` will create the directory along with the necessary parent directories.
Additionally, you can specify a suffix to be attached to the names of each summary file
using using the `--suffix` flag.

```bash
uv run summarize-cli --suffix summary-concise article01.pdf
```

The default is `summary`. You can also specify a summary type with the `--summary-type`
flag. The default is `concise`.

```bash
run summarize-cli --summary-type detailed article01.pdf
```

## Options

Below is a table containing all of the available options:

| Option                 | Description                                                 | Values                                             |
| ---------------------- | ----------------------------------------------------------- | -------------------------------------------------- |
| `--summary-type`, `-s` | The type of summary that will be generated                  | `concise`, `bullet_point`, `detailed`              |
| `--output-dir`, `-d`   | Path to the directory in which the summaries will be stored | Any valid path                                     |
| `--suffix`             | Suffix to use for the output files                          | Any string that will form part of a valid filename |
