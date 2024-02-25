#!/usr/bin/env python

import fileinput
import logging
import os
from typing import Optional

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.llms import LlamaCpp, OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

_GENERATION_TEMPLATE = '''\
Instruction:
- Analyze the provided `git diff` output.
- Generate {n_output_mssage} succinct, informative Git commit messages summarizing the changes.
- Capture the essence and impact of the changes across files.

Output format:
- Present commit messages in a bullet-point list of Markdown.

Input Git Diff result:
```
{input_text}
```
'''     # noqa: E501


def generate_commit_message_from_diff(
    git_diff_txt_path: str, n_output_messages: int = 5,
    llama_model_file_path: Optional[str] = None,
    google_model_name: Optional[str] = 'gemini-pro',
    google_api_key: Optional[str] = None,
    openai_model_name: Optional[str] = 'gpt-3.5-turbo',
    openai_api_key: Optional[str] = None,
    openai_organization: Optional[str] = None, temperature: float = 0.8,
    top_p: float = 0.95, max_tokens: int = 256, n_ctx: int = 512,
    seed: int = -1, token_wise_streaming: bool = False
) -> None:
    '''Extract JSON from input text.'''
    logger = logging.getLogger(__name__)
    if not git_diff_txt_path:
        raise ValueError('The input text file path is not provided.')
    elif llama_model_file_path:
        llm = _read_llm_file(
            path=llama_model_file_path, temperature=temperature, top_p=top_p,
            max_tokens=max_tokens, n_ctx=n_ctx, seed=seed,
            token_wise_streaming=token_wise_streaming
        )
    else:
        overrided_env_vars = {
            'GOOGLE_API_KEY': google_api_key, 'OPENAI_API_KEY': openai_api_key,
            'OPENAI_ORGANIZATION': openai_organization
        }
        for k, v in overrided_env_vars.items():
            if v:
                logger.info(f'Override environment variable: {k}')
                os.environ[k] = v
        if google_model_name:
            llm = ChatGoogleGenerativeAI(
                model=google_model_name
            )   # type: ignore
        else:
            llm = OpenAI(model_name=openai_model_name)
    llm_chain = _create_llm_chain(llm=llm, n_output_mssage=n_output_messages)
    input_text = _read_git_diff_txt(path=git_diff_txt_path)
    logger.info('Genaerating commit messages from the input text.')
    output_string = llm_chain.invoke({'input_text': input_text})
    logger.info(f'LLM output: {output_string}')
    if not output_string:
        raise RuntimeError('LLM output is empty.')
    else:
        print(output_string)


def _create_llm_chain(llm: LlamaCpp, n_output_mssage: int = 5) -> LLMChain:
    logger = logging.getLogger(__name__)
    prompt = PromptTemplate(
        template=_GENERATION_TEMPLATE, input_variables=['input_text'],
        partial_variables={'n_output_mssage': n_output_mssage}
    )
    chain = prompt | llm | StrOutputParser()
    logger.info(f'LLM chain: {chain}')
    return chain


def _read_git_diff_txt(path: str) -> str:
    logger = logging.getLogger(__name__)
    if path == '-':
        logger.info('Read stdin.')
    else:
        logger.info(f'Read a text file: {path}')
    git_diff_txt = ''.join(fileinput.input(files=path))
    logger.debug(f'git_diff_txt: {git_diff_txt}')
    if not git_diff_txt:
        raise ValueError('The input text is empty.')
    else:
        return git_diff_txt


def _read_llm_file(
    path: str, temperature: float = 0.8, top_p: float = 0.95,
    max_tokens: int = 256, n_ctx: int = 512, seed: int = -1,
    token_wise_streaming: bool = False
) -> LlamaCpp:
    logger = logging.getLogger(__name__)
    logger.info(f'Read a Llama 2 model file: {path}')
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
