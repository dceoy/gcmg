#!/usr/bin/env python
# pyright: reportMissingImports=false
# pyright: reportUnknownVariableType=false
# pyright: reportPrivateUsage=false
# pyright: reportUnknownArgumentType=false

import ctypes
import io
import logging
import os

import pytest
from pytest_mock import MockerFixture

from gcmg.llm import (
    _limit_max_tokens,
    _llama_log_callback,
    _read_llm_file,
    create_llm_instance,
)


def test_create_llm_instance_with_model_file(mocker: MockerFixture) -> None:
    llamacpp_model_file_path = "/path/to/model"
    temperature = 0.8
    top_p = 0.95
    max_tokens = 8192
    n_ctx = 512
    seed = -1
    n_batch = 8
    n_gpu_layers = -1
    token_wise_streaming = False
    mocker.patch("gcmg.llm.override_env_vars")
    llm = mocker.MagicMock()
    mock_read_llm_file = mocker.patch("gcmg.llm._read_llm_file", return_value=llm)

    result = create_llm_instance(
        llamacpp_model_file_path=llamacpp_model_file_path,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n_ctx=n_ctx,
        seed=seed,
        n_batch=n_batch,
        n_gpu_layers=n_gpu_layers,
        token_wise_streaming=token_wise_streaming,
    )
    assert result == llm
    mock_read_llm_file.assert_called_once_with(
        path=llamacpp_model_file_path,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n_ctx=n_ctx,
        seed=seed,
        n_batch=n_batch,
        n_gpu_layers=n_gpu_layers,
        token_wise_streaming=token_wise_streaming,
    )


def test_create_llm_instance_with_groq(mocker: MockerFixture) -> None:
    groq_model_name = "dummy-groq-model"
    temperature = 0.8
    max_tokens = 8192
    timeout = None
    max_retries = 2
    stop_sequences = None
    mocker.patch("gcmg.llm.override_env_vars")
    llm = mocker.MagicMock()
    mock_chat_groq = mocker.patch("gcmg.llm.ChatGroq", return_value=llm)

    result = create_llm_instance(
        groq_model_name=groq_model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
    )
    assert result == llm
    mock_chat_groq.assert_called_once_with(
        model=groq_model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        stop_sequences=stop_sequences,
    )


def test_create_llm_instance_with_bedrock(mocker: MockerFixture) -> None:
    bedrock_model_id = "dummy-bedrock-model"
    temperature = 0.8
    max_tokens = 8192
    aws_credentials_profile_name = None
    aws_region = "us-east-1"
    bedrock_endpoint_base_url = "https://api.bedrock.com"
    mocker.patch("gcmg.llm.override_env_vars")
    mocker.patch("gcmg.llm.has_aws_credentials")
    llm = mocker.MagicMock()
    mock_chat_bedrock_converse = mocker.patch(
        "gcmg.llm.ChatBedrockConverse",
        return_value=llm,
    )

    result = create_llm_instance(
        bedrock_model_id=bedrock_model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        aws_credentials_profile_name=aws_credentials_profile_name,
        aws_region=aws_region,
        bedrock_endpoint_base_url=bedrock_endpoint_base_url,
    )
    assert result == llm
    mock_chat_bedrock_converse.assert_called_once_with(
        model=bedrock_model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        region_name=aws_region,
        base_url=bedrock_endpoint_base_url,
        credentials_profile_name=aws_credentials_profile_name,
    )


def test_create_llm_instance_with_google(mocker: MockerFixture) -> None:
    google_model_name = "dummy-google-model"
    temperature = 0.8
    top_p = 0.95
    max_tokens = 8192
    timeout = None
    max_retries = 2
    mocker.patch("gcmg.llm.override_env_vars")
    llm = mocker.MagicMock()
    mock_chat_google_generative_ai = mocker.patch(
        "gcmg.llm.ChatGoogleGenerativeAI",
        return_value=llm,
    )

    result = create_llm_instance(
        google_model_name=google_model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
    )
    assert result == llm
    mock_chat_google_generative_ai.assert_called_once_with(
        model=google_model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
    )


def test_create_llm_instance_with_openai(mocker: MockerFixture) -> None:
    openai_model_name = "dummy-openai-model"
    openai_api_base = "https://api.openai.com"
    openai_organization = "dummy-organization"
    temperature = 0.8
    top_p = 0.95
    seed = -1
    max_tokens = 8192
    timeout = None
    max_retries = 2
    mocker.patch("gcmg.llm.override_env_vars")
    llm = mocker.MagicMock()
    mock_chat_openai = mocker.patch("gcmg.llm.ChatOpenAI", return_value=llm)

    result = create_llm_instance(
        openai_model_name=openai_model_name,
        openai_api_base=openai_api_base,
        openai_organization=openai_organization,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
    )
    assert result == llm
    mock_chat_openai.assert_called_once_with(
        model=openai_model_name,
        base_url=openai_api_base,
        organization=openai_organization,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        max_completion_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
    )


def test_create_llm_instance_no_model_specified(mocker: MockerFixture) -> None:
    mocker.patch("gcmg.llm.override_env_vars")
    mocker.patch.dict(os.environ, {}, clear=True)
    mocker.patch("gcmg.llm.has_aws_credentials", return_value=False)
    with pytest.raises(RuntimeError, match=r"The model cannot be determined."):
        create_llm_instance()


@pytest.mark.parametrize(
    ("max_tokens", "model_name", "default_max_tokens", "expected"),
    [
        (1024, "a", {"a": 512}, 512),
        (1024, "a", {"a": 8192}, 1024),
        (8192, "b", {"a": 512}, 8192),
    ],
)
def test__limit_max_tokens(
    max_tokens: int,
    model_name: str,
    default_max_tokens: dict[str, int],
    expected: int,
    mocker: MockerFixture,
) -> None:
    mock_logger = mocker.MagicMock()
    mocker.patch("logging.getLogger", return_value=mock_logger)
    mocker.patch("gcmg.llm._DEFAULT_MAX_TOKENS", new=default_max_tokens)
    result = _limit_max_tokens(max_tokens=max_tokens, model_name=model_name)
    assert result == expected
    if max_tokens > default_max_tokens.get(model_name, max_tokens):
        mock_logger.warning.assert_called_once()
    else:
        mock_logger.warning.assert_not_called()


@pytest.mark.parametrize(
    ("token_wise_streaming", "logging_level", "expected_verbose"),
    [
        (False, logging.INFO, False),
        (False, logging.DEBUG, True),
        (True, logging.INFO, True),
    ],
)
def test__read_llm_file(
    token_wise_streaming: bool,
    logging_level: int,
    expected_verbose: bool,
    mocker: MockerFixture,
) -> None:
    llm_file_path = "llm.gguf"
    temperature = 0.8
    top_p = 0.95
    max_tokens = 256
    n_ctx = 512
    seed = -1
    n_batch = 8
    n_gpu_layers = -1
    mock_logger = mocker.MagicMock()
    mocker.patch("logging.getLogger", return_value=mock_logger)
    mock_logger.level = logging_level
    mocker.patch("gcmg.llm.llama_log_set")
    expected_result = mocker.Mock()
    mock_llamacpp = mocker.patch("gcmg.llm.LlamaCpp", return_value=expected_result)
    mocker.patch("gcmg.llm.StreamingStdOutCallbackHandler")
    mock_callback_manager = mocker.MagicMock()
    mocker.patch(
        "gcmg.llm.CallbackManager",
        return_value=mock_callback_manager,
    )

    result = _read_llm_file(
        path=llm_file_path,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n_ctx=n_ctx,
        seed=seed,
        n_batch=n_batch,
        n_gpu_layers=n_gpu_layers,
        token_wise_streaming=token_wise_streaming,
    )
    assert result == expected_result
    if token_wise_streaming:
        mock_llamacpp.assert_called_once_with(
            model_path=llm_file_path,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n_ctx=n_ctx,
            seed=seed,
            n_batch=n_batch,
            n_gpu_layers=n_gpu_layers,
            verbose=expected_verbose,
            callback_manager=mock_callback_manager,
        )
    else:
        mock_llamacpp.assert_called_once_with(
            model_path=llm_file_path,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n_ctx=n_ctx,
            seed=seed,
            n_batch=n_batch,
            n_gpu_layers=n_gpu_layers,
            verbose=expected_verbose,
            callback_manager=None,
        )


@pytest.mark.parametrize(
    ("text", "root_level", "expected_output"),
    [
        (b"info message", logging.DEBUG, "info message"),
        (b"debug message", logging.INFO, "debug message"),
        (b"warning message", logging.WARNING, ""),
    ],
)
def test__llama_log_callback(
    text: bytes,
    root_level: int,
    expected_output: str,
    mocker: MockerFixture,
) -> None:
    mocker.patch.object(logging.root, "level", root_level)
    mock_stderr = mocker.patch("gcmg.llm.sys.stderr", new_callable=io.StringIO)
    _llama_log_callback(0, text, ctypes.c_void_p(0))
    assert mock_stderr.getvalue() == expected_output
