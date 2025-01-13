#!/usr/bin/env python
# pyright: reportMissingImports=false
# pyright: reportUnknownVariableType=false
# pyright: reportPrivateUsage=false
# pyright: reportUnknownArgumentType=false

import pytest
from pytest_mock import MockerFixture

from gcmg.suggestion import (
    _GENERATION_TEMPLATE,
    _generate_and_print_commit_messages,
    _read_git_diff_txt,
    generate_commit_message_from_diff,
)


def test_generate_commit_message_from_diff(mocker: MockerFixture) -> None:
    git_diff_txt_path = "input.txt"
    n_output_messages = 5
    llamacpp_model_file_path = "model.gguf"
    groq_model_name = "groq_model"
    bedrock_model_id = "bedrock_model"
    google_model_name = "google_model"
    openai_model_name = "openai_model"
    temperature = 0.8
    top_p = 0.95
    max_tokens = 8192
    n_ctx = 512
    seed = -1
    n_batch = 8
    n_gpu_layers = -1
    token_wise_streaming = False
    timeout = None
    max_retries = 2
    git = "/usr/bin/git"
    mock_llm_instance = mocker.MagicMock()
    mock_create_llm_instance = mocker.patch(
        "gcmg.suggestion.create_llm_instance",
        return_value=mock_llm_instance,
    )
    dummy_diff_txt = "dummy diff"
    mock__read_git_diff_txt = mocker.patch(
        "gcmg.suggestion._read_git_diff_txt",
        return_value=dummy_diff_txt,
    )
    mock__generate_and_print_commit_messages = mocker.patch(
        "gcmg.suggestion._generate_and_print_commit_messages",
    )
    generate_commit_message_from_diff(
        git_diff_txt_path=git_diff_txt_path,
        n_output_messages=n_output_messages,
        llamacpp_model_file_path=llamacpp_model_file_path,
        groq_model_name=groq_model_name,
        bedrock_model_id=bedrock_model_id,
        google_model_name=google_model_name,
        openai_model_name=openai_model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n_ctx=n_ctx,
        seed=seed,
        n_batch=n_batch,
        n_gpu_layers=n_gpu_layers,
        token_wise_streaming=token_wise_streaming,
        timeout=timeout,
        max_retries=max_retries,
        git=git,
    )
    mock_create_llm_instance.assert_called_once_with(
        llamacpp_model_file_path=llamacpp_model_file_path,
        groq_model_name=groq_model_name,
        groq_api_key=None,
        bedrock_model_id=bedrock_model_id,
        google_model_name=google_model_name,
        google_api_key=None,
        openai_model_name=openai_model_name,
        openai_api_key=None,
        openai_api_base=None,
        openai_organization=None,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n_ctx=n_ctx,
        seed=seed,
        n_batch=n_batch,
        n_gpu_layers=n_gpu_layers,
        token_wise_streaming=token_wise_streaming,
        timeout=timeout,
        max_retries=max_retries,
        aws_credentials_profile_name=None,
        aws_region=None,
        bedrock_endpoint_base_url=None,
    )
    mock__read_git_diff_txt.assert_called_once_with(path=git_diff_txt_path, git=git)
    mock__generate_and_print_commit_messages.assert_called_once_with(
        git_diff_text=dummy_diff_txt,
        llm=mock_llm_instance,
        n_output_messages=n_output_messages,
    )


@pytest.mark.parametrize("git_diff_txt", [(None), (""), ("dummy diff")])
def test__generate_and_print_commit_messages(
    git_diff_txt: str | None,
    mocker: MockerFixture,
) -> None:
    mock_logger = mocker.MagicMock()
    mocker.patch("logging.getLogger", return_value=mock_logger)
    n_output_messages = 5
    llm_output = "dummy output"
    mock_llm_chain = mocker.MagicMock()
    mock_prompt_template = mocker.patch(
        "gcmg.suggestion.PromptTemplate",
        return_value=mock_llm_chain,
    )
    mocker.patch("gcmg.suggestion.StrOutputParser", return_value=mock_llm_chain)
    mock_llm_chain.__or__.return_value = mock_llm_chain
    mock_llm_chain.invoke.return_value = llm_output
    mock_print = mocker.patch("gcmg.suggestion.print")

    _generate_and_print_commit_messages(
        git_diff_text=git_diff_txt,
        llm=mock_llm_chain,
        n_output_messages=n_output_messages,
    )
    if not git_diff_txt:
        mock_logger.warning.assert_called_once_with("Git diff result is empty.")
        mock_llm_chain.invoke.assert_not_called()
    else:
        mock_prompt_template.assert_called_once_with(
            template=_GENERATION_TEMPLATE,
            input_variables=["input_text"],
            partial_variables={"n_output_messages": str(n_output_messages)},
        )
        mock_llm_chain.invoke.assert_called_once_with({"input_text": git_diff_txt})
        mock_print.assert_called_once_with(llm_output)


def test__generate_and_print_commit_messages_with_empty_llm_output(
    mocker: MockerFixture,
) -> None:
    mocker.patch("logging.getLogger")
    mock_llm_chain = mocker.MagicMock()
    mocker.patch("gcmg.suggestion.PromptTemplate", return_value=mock_llm_chain)
    mocker.patch("gcmg.suggestion.StrOutputParser", return_value=mock_llm_chain)
    mock_llm_chain.__or__.return_value = mock_llm_chain
    mock_llm_chain.invoke.return_value = ""

    with pytest.raises(RuntimeError, match=r"LLM output is empty."):
        _generate_and_print_commit_messages(
            git_diff_text="dummy diff",
            llm=mock_llm_chain,
            n_output_messages=5,
        )


@pytest.mark.parametrize(
    ("path", "git"),
    [
        (None, "git"),
        ("-", "git"),
        ("input.txt", "/usr/bin/git"),
    ],
)
def test__read_git_diff_txt(path: str | None, git: str, mocker: MockerFixture) -> None:
    mocker.patch("logging.getLogger")
    output_txt = "dummy output"
    mock_fileinput_input = mocker.patch(
        "gcmg.suggestion.fileinput.input",
        return_value=output_txt,
    )
    subprocess_run_output = mocker.MagicMock()
    subprocess_run_output.stdout.decode.return_value = output_txt
    mock_subprocess_run = mocker.patch(
        "gcmg.suggestion.subprocess.run",
        return_value=subprocess_run_output,
    )

    git_diff_txt = _read_git_diff_txt(path=path, git=git)
    assert git_diff_txt == output_txt
    if path:
        mock_fileinput_input.assert_called_once_with(files=path)
        mock_subprocess_run.assert_not_called()
    else:
        mock_fileinput_input.assert_not_called()
        mock_subprocess_run.assert_called_once_with(
            [git, "diff", "--staged"],
            capture_output=True,
            check=True,
        )
