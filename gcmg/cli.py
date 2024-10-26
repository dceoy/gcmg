#!/usr/bin/env python
"""Command-line tool to generate git commit messages from git diff output."""

import typer
from rich import print
from typing_extensions import Annotated

from . import __version__
from .suggestion import generate_commit_message_from_diff
from .utility import configure_logging

app = typer.Typer()


def _version_callback(value: bool) -> None:
    if value:
        print(__version__)
        raise typer.Exit


@app.command()
def main(
    git_diff_txt_path: Annotated[
        str | None,
        typer.Argument(help='Git diff output text file or "-" for stdin.'),
    ] = None,
    n_output_messages: Annotated[
        int,
        typer.Option(
           "--n-output-messages", help="Specify the number of output messages.",
        ),
    ] = 5,
    temperature: Annotated[
        float,
        typer.Option("--temperature", help="Specify the temperature for sampling."),
    ] = 0,
    top_p: Annotated[
        float, typer.Option("--top-p", help="Specify the top-p value for sampling."),
    ] = 0.1,
    max_tokens: Annotated[
        int, typer.Option("--max-tokens", help="Specify the max tokens to generate."),
    ] = 8000,
    n_ctx: Annotated[
        int, typer.Option("--n-ctx", help="Specify the token context window."),
    ] = 1024,
    seed: Annotated[int, typer.Option("--seed", help="Specify the random seed.")] = -1,
    n_batch: Annotated[
        int, typer.Option("--n-batch", help="Specify the number of batch tokens."),
    ] = 8,
    n_gpu_layers: Annotated[
        int, typer.Option("--n-gpu-layers", help="Specify the number of GPU layers."),
    ] = -1,
    openai_model_name: Annotated[
        str | None,
        typer.Option(
            "--openai-model",
            envvar="OPENAI_MODEL",
            help="Use the OpenAI model. (e.g., gpt-4o-mini)",
        ),
    ] = None,
    google_model_name: Annotated[
        str | None,
        typer.Option(
            "--google-model",
            envvar="GOOGLE_MODEL",
            help="Use the Google Generative AI model. (e.g., gemini-1.5-flash)",
        ),
    ] = None,
    groq_model_name: Annotated[
        str | None,
        typer.Option(
            "--groq-model",
            envvar="GROQ_MODEL",
            help="Use the Groq model. (e.g., llama-3.1-70b-versatile)",
        ),
    ] = None,
    bedrock_model_id: Annotated[
        str | None,
        typer.Option(
            "--bedrock-model",
            envvar="BEDROCK_MODEL",
            help=(
                "Use the Amazon Bedrock model."
                " (e.g., anthropic.claude-3-5-sonnet-20240620-v1:0)"
            ),
        ),
    ] = None,
    llamacpp_model_file_path: Annotated[
        str | None,
        typer.Option("--model-file", help="Use the model GGUF file for llama.cpp."),
    ] = None,
    openai_api_key: Annotated[
        str | None,
        typer.Option(
            "--openai-api-key",
            envvar="OPENAI_API_KEY",
            help="Override the OpenAI API key.",
        ),
    ] = None,
    openai_api_base: Annotated[
        str | None,
        typer.Option(
            "--openai-api-base",
            envvar="OPENAI_API_BASE",
            help="Override the OpenAI API base URL.",
        ),
    ] = None,
    openai_organization: Annotated[
        str | None,
        typer.Option(
            "--openai-organization",
            envvar="OPENAI_ORGANIZATION",
            help="Override the OpenAI organization ID.",
        ),
    ] = None,
    google_api_key: Annotated[
        str | None,
        typer.Option(
            "--google-api-key",
            envvar="GOOGLE_API_KEY",
            help="Override the Google API key.",
        ),
    ] = None,
    groq_api_key: Annotated[
        str | None,
        typer.Option(
            "--groq-api-key",
            envvar="GROQ_API_KEY",
            help="Override the Groq API key.",
        ),
    ] = None,
    debug: Annotated[
        bool, typer.Option("--debug", help="Execute with debug messages."),
    ] = False,
    info: Annotated[
        bool, typer.Option("--info", help="Execute with info messages."),
    ] = False,
    version: Annotated[     # noqa: ARG001
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show version information and exit.",
        ),
    ] = False,
) -> None:
    """Suggest commit messages from git diff output."""
    configure_logging(debug=debug, info=info)
    generate_commit_message_from_diff(
        git_diff_txt_path=git_diff_txt_path,
        n_output_messages=n_output_messages,
        llamacpp_model_file_path=llamacpp_model_file_path,
        groq_model_name=groq_model_name,
        groq_api_key=groq_api_key,
        bedrock_model_id=bedrock_model_id,
        google_model_name=google_model_name,
        google_api_key=google_api_key,
        openai_model_name=openai_model_name,
        openai_api_key=openai_api_key,
        openai_api_base=openai_api_base,
        openai_organization=openai_organization,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n_ctx=n_ctx,
        seed=seed,
        n_batch=n_batch,
        n_gpu_layers=n_gpu_layers,
    )
