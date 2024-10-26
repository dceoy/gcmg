#!/usr/bin/env python
"""Utility functions."""

import logging
import os

import boto3
from botocore.exceptions import NoCredentialsError
from mypy_boto3_sts.client import STSClient


def configure_logging(
    debug: bool = False,
    info: bool = False,
    format: str = "%(asctime)s [%(levelname)-8s] <%(name)s> %(message)s",
) -> None:
    """Configure the logging module.

    Args:
        debug: Enable the debug level.
        info: Enable the info level.
        format: The format of the log message.
    """
    if debug:
        lv = logging.DEBUG
    elif info:
        lv = logging.INFO
    else:
        lv = logging.WARNING
    logging.basicConfig(format=format, level=lv)


def has_aws_credentials() -> bool:
    """Check if the AWS credentials are available.

    Returns:
        True if the AWS credentials are available, False otherwise.
    """
    logger = logging.getLogger(has_aws_credentials.__name__)
    sts: STSClient = boto3.client("sts")  # pyright: ignore[reportUnknownMemberType]
    try:
        caller_identity = sts.get_caller_identity()
    except NoCredentialsError as e:
        logger.debug("caller_identity: %s", e)
        return False
    else:
        logger.debug("caller_identity: %s", caller_identity)
        return True


def override_env_vars(**kwargs: str | None) -> None:
    """Override the environment variables.

    Args:
        kwargs: The key-value pairs of the environment variables to be overridden.
    """
    logger = logging.getLogger(override_env_vars.__name__)
    for k, v in kwargs.items():
        if v is not None:
            logger.info("Override the environment variable: %s=%s", k, v)
            os.environ[k] = v
        else:
            logger.info("Skip to override environment variable: %s", k)
