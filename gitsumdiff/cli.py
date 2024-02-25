#!/usr/bin/env python
"""
Command-line tool to generate git commit messages from git diff output

Usage:
    git-sum-diff [--debug|--info] [--temperature=<float>] [--top-p=<float>]
        [--max-tokens=<int>] [--n-ctx=<int>] [--seed=<int>]
        [--openai-model=<name>|--google-model=<name>|--llama-model-gguf=<path>]
        [--openai-api-key=<str>] [--openai-organization=<str>]
        [--google-api-key=<str>] [--n-output-messages=<int>] [<git_diff_txt>]
    git-sum-diff -h|--help
    git-sum-diff --version

Options:
    --debug, --info         Execute a command with debug|info messages
    --temperature=<float>   Specify the temperature for sampling [default: 0]
    --top-p=<float>         Specify the top-p value for sampling [default: 0.1]
    --max-tokens=<int>      Specify the max tokens to generate [default: 8192]
    --n-ctx=<int>           Specify the token context window [default: 1024]
    --seed=<int>            Specify the random seed [default: -1]
    --openai-model=<name>   Use the OpenAI model (e.g., gpt-3.5-turbo)
                            This option requires the environment variables:
                            - OPENAI_API_KEY (OpenAI API key)
                            - OPENAI_ORGANIZATION (OpenAI organization ID)
    --google-model=<name>   Use the Google model (e.g., gemini-pro)
                            This option requires the environment variable:
                            - GOOGLE_API_KEY (Google API key)
    --llama-model-gguf=<path>
                            Use the LLaMA model GGUF file
    --openai-api-key=<str>  Override the OpenAI API key ($OPENAI_API_KEY)
    --openai-organization=<str>
                            Override the OpenAI organization ID
                            ($OPENAI_ORGANIZATION)
    --google-api-key=<str>  Override the Google API key ($GOOGLE_API_KEY)
    --n-output-messages=<int>
                            Specify the number of output messages [default: 5]
    -h, --help              Print help and exit
    --version               Print version and exit

Arguments:
    <git_diff_txt>          Git diff output text file or "-" for stdin
"""

import logging
import os
import signal

from docopt import docopt

from . import __version__
from .llm import generate_commit_message_from_diff


def main():
    args = docopt(__doc__, version=__version__)
    _set_log_config(debug=args['--debug'], info=args['--info'])
    logger = logging.getLogger(__name__)
    logger.debug(f'args:{os.linesep}{args}')
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    either_required_args = [
        '--openai-model', '--google-model', '--llama-model-gguf'
    ]
    if not [s for s in either_required_args if args[s]]:
        raise ValueError(
            'Either one of the following options is required: '
            + ', '.join(either_required_args)
        )
    else:
        generate_commit_message_from_diff(
            git_diff_txt_path=args['<git_diff_txt>'],
            n_output_messages=int(args['--n-output-messages']),
            llama_model_file_path=args['--llama-model-gguf'],
            google_model_name=args['--google-model'],
            google_api_key=args['--google-api-key'],
            openai_model_name=args['--openai-model'],
            openai_api_key=args['--openai-api-key'],
            openai_organization=args['--openai-organization'],
            temperature=float(args['--temperature']),
            top_p=float(args['--top-p']),
            max_tokens=int(args['--max-tokens']),
            n_ctx=int(args['--n-ctx']), seed=int(args['--seed'])
        )


def _set_log_config(debug=None, info=None):
    if debug:
        lv = logging.DEBUG
    elif info:
        lv = logging.INFO
    else:
        lv = logging.WARNING
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S', level=lv
    )
