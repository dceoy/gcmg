#!/usr/bin/env python

import fileinput
import logging
import os
import subprocess  # nosec B404
from typing import Optional

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.llms import LlamaCpp
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

_GENERATION_TEMPLATE = '''\
Instruction:
- Analyze the provided `git diff` output.
- Generate {n_output_mssage} succinct, informative Git commit messages summarizing the changes.
- Capture the essence and impact of the changes across files.

Output format:
- Present commit messages in a bullet-point list of Markdown.

Input `git diff` result:
```
{input_text}
```
'''     # noqa: E501


def generate_commit_message_from_diff(
    git_diff_txt_path: str, n_output_messages: str = '5',
    llama_model_file_path: Optional[str] = None,
    google_model_name: Optional[str] = 'gemini-pro',
    google_api_key: Optional[str] = None,
    openai_model_name: Optional[str] = 'gpt-3.5-turbo',
    openai_api_key: Optional[str] = None,
    openai_organization: Optional[str] = None, temperature: float = 0.8,
    top_p: float = 0.95, max_tokens: int = 256, n_ctx: int = 512,
    seed: int = -1, token_wise_streaming: bool = False, git: str = 'git'
) -> None:
    '''Extract JSON from input text.'''
    logger = logging.getLogger(__name__)
    if llama_model_file_path:
        llm = _read_llm_file(
            path=llama_model_file_path, temperature=temperature, top_p=top_p,
            max_tokens=max_tokens, n_ctx=n_ctx, seed=seed,
            token_wise_streaming=token_wise_streaming
        )
    elif google_model_name:
        _override_env_vars(GOOGLE_API_KEY=google_api_key)
        logger.info(f'Use the Google model: {google_model_name}')
        llm = ChatGoogleGenerativeAI(model=google_model_name)   # type: ignore
    else:
        _override_env_vars(
            OPENAI_API_KEY=openai_api_key,
            OPENAI_ORGANIZATION=openai_organization
        )
        logger.info(f'Use the OpenAI model: {openai_model_name}')
        llm = ChatOpenAI(model_name=openai_model_name)  # type: ignore
    llm_chain = _create_llm_chain(llm=llm, n_output_mssage=n_output_messages)
    input_text = _read_git_diff_txt(path=git_diff_txt_path, git=git)
    if not input_text:
        logger.warning('Git diff result is empty.')
    else:
        logger.info('Genaerating commit messages from the input text.')
        output_string = llm_chain.invoke({'input_text': input_text})
        logger.debug(f'LLM output: {output_string}')
        if not output_string:
            raise RuntimeError('LLM output is empty.')
        else:
            print(output_string)


def _create_llm_chain(llm: LlamaCpp, n_output_mssage: str = '5') -> LLMChain:
    logger = logging.getLogger(__name__)
    prompt = PromptTemplate(
        template=_GENERATION_TEMPLATE, input_variables=['input_text'],
        partial_variables={'n_output_mssage': n_output_mssage}
    )
    chain = prompt | llm | StrOutputParser()
    logger.debug(f'LLM chain: {chain}')
    return chain


def _read_git_diff_txt(path: Optional[str] = None, git: str = 'git') -> str:
    logger = logging.getLogger(__name__)
    if path == '-':
        logger.info('Read stdin.')
        git_diff_txt = ''.join(fileinput.input(files=path))
    elif path:
        logger.info(f'Read a text file: {path}')
        git_diff_txt = ''.join(fileinput.input(files=path))
    else:
        cmd = f'{git} diff HEAD'
        logger.info(f'Read a result of `{cmd}`.')
        git_diff = subprocess.run(
            cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True
        )   # nosec B603
        if git_diff.returncode == 0:
            git_diff_txt = git_diff.stdout
        else:
            raise RuntimeError(f'Failed to execute `{cmd}`: {git_diff.stderr}')
    logger.debug(f'git_diff_txt: {git_diff_txt}')
    return git_diff_txt


def _override_env_vars(**kwargs: Optional[str]) -> None:
    logger = logging.getLogger(__name__)
    for k, v in kwargs.items():
        if v:
            logger.info(f'Override environment variable: {k}')
            os.environ[k] = v


def _read_llm_file(
    path: str, temperature: float = 0.8, top_p: float = 0.95,
    max_tokens: int = 256, n_ctx: int = 512, seed: int = -1,
    token_wise_streaming: bool = False
) -> LlamaCpp:
    logger = logging.getLogger(__name__)
    logger.info(f'Read a Llama model file: {path}')
    llm = LlamaCpp(
        model_path=path, temperature=temperature, top_p=top_p,
        max_tokens=max_tokens, n_ctx=n_ctx, seed=seed,
        verbose=(
            token_wise_streaming or logging.getLogger().level <= logging.DEBUG
        ),
        callback_manager=(
            CallbackManager([StreamingStdOutCallbackHandler()])
            if token_wise_streaming else None
        )
    )
    logger.debug(f'llm: {llm}')
    return llm
