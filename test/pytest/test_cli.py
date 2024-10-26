#!/usr/bin/env python
# pyright: reportPrivateUsage=false

import pytest
from pytest_mock import MockerFixture
from typer import Exit
from typer.testing import CliRunner

from gcmg.cli import _version_callback, app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.mark.parametrize("value", [True, False])
def test__version_callback(value: bool, mocker: MockerFixture) -> None:
    dummy_version = "1.0.0"
    mocker.patch("gcmg.cli.__version__", dummy_version)
    mock_print = mocker.patch("gcmg.cli.print")
    if value:
        with pytest.raises(Exit):
            _version_callback(value)
        mock_print.assert_called_once_with(dummy_version)
    else:
        _version_callback(False)
        mock_print.assert_not_called()


def test_main_with_help_option(runner: CliRunner) -> None:
    result = runner.invoke(app, "--help")
    assert result.exit_code == 0
    assert "Usage:" in result.stdout


def test_main_invalid_arguments(runner: CliRunner) -> None:
    result = runner.invoke(app, "--invalid-option")
    assert result.exit_code != 0
    assert "Error" in result.stdout


def test_main_with_version_option(runner: CliRunner, mocker: MockerFixture) -> None:
    dummy_version = "1.0.0"
    mocker.patch("gcmg.cli.__version__", dummy_version)
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert dummy_version in result.stdout


@pytest.mark.parametrize(
    "cli_args",
    [
        [],
        ["-"],
        ["diff.txt"],
        ["--debug"],
        ["--info"],
        [
            "--openai-model=gpt-4o-mini",
            "--openai-api-base=https://example.com/api",
            "--openai-organization=example",
            "--openai-api-key=dummy",
        ],
        ["--google-model=gemini-1.5-flash", "--google-api-key=dummy"],
        ["--groq-model=llama-3.1-70b-versatile", "--groq-api-key=dummy"],
        ["--bedrock-model=anthropic.claude-3-5-sonnet-20240620-v1:0"],
        ["--model-file=llm.gguf"],
        [
            "--model-file=llm.gguf",
            "--temperature=0.5",
            "--top-p=0.2",
            "--max-tokens=4000",
            "--n-ctx=512",
            "--seed=42",
            "--n-batch=4",
            "--n-gpu-layers=2",
        ],
    ],
)
def test_main_command(
    cli_args: list[str],
    runner: CliRunner,
    mocker: MockerFixture,
) -> None:
    mock_configure_logging = mocker.patch("gcmg.cli.configure_logging")
    mock_generate_commit_message_from_diff = mocker.patch(
        "gcmg.cli.generate_commit_message_from_diff",
    )
    result = runner.invoke(app, cli_args)
    assert result.exit_code == 0
    mock_configure_logging.assert_called_once_with(
        debug=("--debug" in cli_args),
        info=("--info" in cli_args),
    )
    mock_generate_commit_message_from_diff.assert_called_once()
