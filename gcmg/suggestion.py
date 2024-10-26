#!/usr/bin/env python
"""Functions to generate commit messages."""

import fileinput
import logging
import subprocess  # noqa: S404

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_aws import ChatBedrockConverse
from langchain_community.llms import LlamaCpp
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from rich import print

from .llm import create_llm_instance

_GENERATION_TEMPLATE = """\
Instruction:
- Analyze the provided `git diff` output.
- Generate {n_output_messages} succinct, informative Git commit messages summarizing the changes.
- Capture the essence and impact of the changes across files.

Output format:
- Present commit messages in a bullet-point list of Markdown.

Input `git diff` result:
```
{input_text}
```
"""  # noqa: E501


def generate_commit_message_from_diff(
    git_diff_txt_path: str | None = None,
    n_output_messages: int = 5,
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
    git: str = "git",
) -> None:
    """Extract JSON from input text.

    Args:
        git_diff_txt_path: Path to the input text file.
        n_output_messages: Number of output messages.
        llamacpp_model_file_path: Path to the LlamaCpp model file.
        groq_model_name: Name of the Groq model.
        groq_api_key: API key
        bedrock_model_id: Bedrock model ID.
        google_model_name: Name of the Google model.
        google_api_key: API key of the Google model.
        openai_model_name: Name of the OpenAI model.
        openai_api_key: API key of the OpenAI model.
        openai_api_base: Base URL of the OpenAI API.
        openai_organization: Organization of the OpenAI.
        temperature: Temperature of the model.
        top_p: Top-p of the model.
        max_tokens: Maximum number of tokens.
        n_ctx: Context size.
        seed: Seed of the model.
        n_batch: Batch size.
        n_gpu_layers: Number of GPU layers.
        token_wise_streaming: Flag to enable token-wise streaming.
        timeout: Timeout of the model.
        max_retries: Maximum number of retries.
        aws_credentials_profile_name: Name of the AWS credentials profile.
        aws_region: AWS region.
        bedrock_endpoint_base_url: Base URL of the Amazon Bedrock endpoint.
        git: Path to the git executable.
    """
    llm = create_llm_instance(
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
        token_wise_streaming=token_wise_streaming,
        timeout=timeout,
        max_retries=max_retries,
        aws_credentials_profile_name=aws_credentials_profile_name,
        aws_region=aws_region,
        bedrock_endpoint_base_url=bedrock_endpoint_base_url,
    )
    git_diff_text = _read_git_diff_txt(path=git_diff_txt_path, git=git)
    _generate_and_print_commit_messages(
        git_diff_text=git_diff_text,
        llm=llm,
        n_output_messages=n_output_messages,
    )


def _generate_and_print_commit_messages(
    git_diff_text: str | None,
    llm: LlamaCpp
    | ChatGroq
    | ChatBedrockConverse
    | ChatGoogleGenerativeAI
    | ChatOpenAI,
    n_output_messages: int = 5,
) -> None:
    logger = logging.getLogger(__name__)
    if not git_diff_text:
        logger.warning("Git diff result is empty.")
    else:
        logger.info("Genaerating commit messages from the input text.")
        prompt = PromptTemplate(
            template=_GENERATION_TEMPLATE,
            input_variables=["input_text"],
            partial_variables={"n_output_messages": str(n_output_messages)},
        )
        llm_chain: LLMChain = prompt | llm | StrOutputParser()
        logger.debug("LLM chain: %s", llm_chain)
        llm_output = llm_chain.invoke({"input_text": git_diff_text})
        logger.debug("LLM output: %s", llm_output)
        if llm_output:
            print(llm_output)
        else:
            raise RuntimeError("LLM output is empty.")


def _read_git_diff_txt(path: str | None = None, git: str = "git") -> str | None:
    logger = logging.getLogger(__name__)
    if path:
        logger.info("Read stdin or a file: %s", path)
        git_diff_txt = "".join(fileinput.input(files=path))  # noqa: SIM115
    else:
        cmd = [git, "diff", "HEAD"]
        logger.info("Read a result of `%s`.", cmd)
        git_diff_txt = subprocess.run(  # noqa: S603
            cmd, capture_output=True, check=True
        ).stdout.decode()
    logger.debug("git_diff_txt: %s", git_diff_txt)
    return git_diff_txt
