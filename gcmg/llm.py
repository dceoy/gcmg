#!/usr/bin/env python
"""Functions for LLM."""

import ctypes
import logging
import os
import sys

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_aws import ChatBedrockConverse
from langchain_community.llms import LlamaCpp
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from llama_cpp import llama_log_callback, llama_log_set

from .utility import has_aws_credentials, override_env_vars

_DEFAULT_MODEL_NAMES = {
    "openai": "gpt-4o-mini",
    "google": "gemini-1.5-flash",
    "groq": "llama-3.1-70b-versatile",
    "bedrock": "anthropic.claude-3-5-sonnet-20240620-v1:0",
}
_DEFAULT_MAX_TOKENS = {
    "gpt-4o": 128000,
    "gpt-4o-2024-05-13": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4o-mini-2024-07-18": 128000,
    "gpt-4o-2024-08-06": 128000,
    "o1-mini": 128000,
    "o1-mini-2024-09-12": 128000,
    "o1-preview": 128000,
    "o1-preview-2024-09-12": 128000,
    "claude-3-5-sonnet@20240620": 100000,
    "gemini-1.5-pro": 1048576,
    "gemini-1.5-flash": 1048576,
    "gemma2": 8200,
    "gemma2-9b-it": 8192,
    "claude-3-5-sonnet": 100000,
    "claude-3-5-sonnet-20240620": 100000,
    "anthropic.claude-3-5-sonnet-20240620-v1:0": 100000,
    "mixtral-8x7b-32768": 32768,
    "llama-3.1-8b-instant": 131072,
    "llama-3.1-70b-versatile": 131072,
    "llama-3.1-405b-reasoning": 131072,
}


def create_llm_instance(
    llamacpp_model_file_path: str | None = None,
    groq_model_name: str | None = None,
    groq_api_key: str | None = None,
    bedrock_model_id: str | None = None,
    google_model_name: str | None = None,
    google_api_key: str | None = None,
    openai_model_name: str | None = None,
    openai_api_key: str | None = None,
    openai_api_base: str | None = None,
    openai_organization: str | None = None,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_tokens: int = 8192,
    n_ctx: int = 512,
    seed: int = -1,
    n_batch: int = 8,
    n_gpu_layers: int = -1,
    token_wise_streaming: bool = False,
    timeout: int | None = None,
    max_retries: int = 2,
    aws_credentials_profile_name: str | None = None,
    aws_region: str | None = None,
    bedrock_endpoint_base_url: str | None = None,
) -> LlamaCpp | ChatGroq | ChatBedrockConverse | ChatGoogleGenerativeAI | ChatOpenAI:
    """Create an instance of LLM.

    Args:
        llamacpp_model_file_path: The file path of the LLM model.
        groq_model_name: The name of the GROQ model.
        groq_api_key: The API
        bedrock_model_id: The ID of the Amazon Bedrock model.
        google_model_name: The name of the Google Generative AI model.
        google_api_key: The API key of the Google Generative AI.
        openai_model_name: The name of the OpenAI model.
        openai_api_key: The API key of the OpenAI.
        openai_api_base: The base URL of the OpenAI API.
        openai_organization: The organization of the OpenAI.
        temperature: The temperature of the model.
        top_p: The top-p of the model.
        max_tokens: The maximum number of tokens.
        n_ctx: The context size.
        seed: The seed of the model.
        n_batch: The batch size.
        n_gpu_layers: The number of GPU layers.
        token_wise_streaming: The flag to enable token-wise streaming.
        timeout: The timeout of the model.
        max_retries: The maximum number of retries.
        aws_credentials_profile_name: The name of the AWS credentials profile.
        aws_region: The AWS region.
        bedrock_endpoint_base_url: The base URL of the Amazon Bedrock endpoint.

    Returns:
        An instance of LLM.

    Raises:
        RuntimeError: The model cannot be determined.
    """
    logger = logging.getLogger(create_llm_instance.__name__)
    override_env_vars(
        GROQ_API_KEY=groq_api_key,
        GOOGLE_API_KEY=google_api_key,
        OPENAI_API_KEY=openai_api_key,
    )
    if llamacpp_model_file_path:
        logger.info("Use local LLM: %s", llamacpp_model_file_path)
        return _read_llm_file(
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
    elif groq_model_name or (
        (not any([bedrock_model_id, google_model_name, openai_model_name]))
        and os.environ.get("GROQ_API_KEY")
    ):
        logger.info("Use GROQ: %s", groq_model_name)
        m = groq_model_name or _DEFAULT_MODEL_NAMES["groq"]
        return ChatGroq(
            model=m,
            temperature=temperature,
            max_tokens=_limit_max_tokens(max_tokens=max_tokens, model_name=m),
            timeout=timeout,
            max_retries=max_retries,
            stop_sequences=None,
        )
    elif bedrock_model_id or (
        (not any([google_model_name, openai_model_name])) and has_aws_credentials()
    ):
        logger.info("Use Amazon Bedrock: %s", bedrock_model_id)
        m = bedrock_model_id or _DEFAULT_MODEL_NAMES["bedrock"]
        return ChatBedrockConverse(
            model=m,
            temperature=temperature,
            max_tokens=_limit_max_tokens(max_tokens=max_tokens, model_name=m),
            region_name=aws_region,
            base_url=bedrock_endpoint_base_url,
            credentials_profile_name=aws_credentials_profile_name,
        )
    elif google_model_name or (
        (not openai_model_name) and os.environ.get("GOOGLE_API_KEY")
    ):
        logger.info("Use Google Generative AI: %s", google_model_name)
        m = google_model_name or _DEFAULT_MODEL_NAMES["google"]
        return ChatGoogleGenerativeAI(
            model=m,
            temperature=temperature,
            top_p=top_p,
            max_tokens=_limit_max_tokens(max_tokens=max_tokens, model_name=m),
            timeout=timeout,
            max_retries=max_retries,
        )
    elif openai_model_name or os.environ.get("OPENAI_API_KEY"):
        logger.info("Use OpenAI: %s", openai_model_name)
        logger.info("OpenAI API base: %s", openai_api_base)
        logger.info("OpenAI organization: %s", openai_organization)
        m = openai_model_name or _DEFAULT_MODEL_NAMES["openai"]
        return ChatOpenAI(
            model=m,
            base_url=openai_api_base,
            organization=openai_organization,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            max_completion_tokens=_limit_max_tokens(
                max_tokens=max_tokens, model_name=m
            ),
            timeout=timeout,
            max_retries=max_retries,
        )
    else:
        raise RuntimeError("The model cannot be determined.")


def _limit_max_tokens(max_tokens: int, model_name: str) -> int:
    default_max_tokens = _DEFAULT_MAX_TOKENS.get(model_name, max_tokens)
    if max_tokens > default_max_tokens:
        logging.getLogger(_limit_max_tokens.__name__).warning(
            "The maximum number of tokens is limited to %d.",
            default_max_tokens,
        )
        return default_max_tokens
    else:
        return max_tokens


def _read_llm_file(
    path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_tokens: int = 256,
    n_ctx: int = 512,
    seed: int = -1,
    n_batch: int = 8,
    n_gpu_layers: int = -1,
    token_wise_streaming: bool = False,
) -> LlamaCpp:
    logger = logging.getLogger(_read_llm_file.__name__)
    llama_log_set(_llama_log_callback, ctypes.c_void_p(0))
    logger.info("Read the model file: %s", path)
    llm = LlamaCpp(
        model_path=path,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n_ctx=n_ctx,
        seed=seed,
        n_batch=n_batch,
        n_gpu_layers=n_gpu_layers,
        verbose=(token_wise_streaming or logger.level <= logging.DEBUG),
        callback_manager=(
            CallbackManager([StreamingStdOutCallbackHandler()])
            if token_wise_streaming
            else None
        ),
    )
    logger.debug("llm: %s", llm)
    return llm


@llama_log_callback
def _llama_log_callback(level: int, text: bytes, user_data: ctypes.c_void_p) -> None:  # noqa: ARG001
    if logging.root.level < logging.WARNING:
        print(text.decode("utf-8"), end="", flush=True, file=sys.stderr)
