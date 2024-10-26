#!/usr/bin/env python

import logging
import os
from typing import Generator

import pytest
from botocore.exceptions import NoCredentialsError
from pytest_mock import MockerFixture

from gcmg.utility import (
    configure_logging,
    has_aws_credentials,
    override_env_vars,
)


@pytest.mark.parametrize(
    ("debug", "info", "expected_level"),
    [
        (True, False, logging.DEBUG),
        (False, True, logging.INFO),
        (False, False, logging.WARNING),
    ],
)
def test_configure_logging(
    debug: bool,
    info: bool,
    expected_level: int,
    mocker: MockerFixture,
) -> None:
    logging_format = "%(asctime)s [%(levelname)-8s] <%(name)s> %(message)s"
    mock_logging_basic_config = mocker.patch("logging.basicConfig")
    configure_logging(debug=debug, info=info, format=logging_format)
    mock_logging_basic_config.assert_called_once_with(
        format=logging_format,
        level=expected_level,
    )


@pytest.mark.parametrize("expected", [(True), (False)])
def test_has_aws_credentials(expected: bool, mocker: MockerFixture) -> None:
    sts_client = mocker.MagicMock()
    mocker.patch("gcmg.utility.boto3.client", return_value=sts_client)
    if expected:
        sts_client.get_caller_identity.return_value = {"Account": "123456789012"}
    else:
        sts_client.get_caller_identity.side_effect = NoCredentialsError()
    assert has_aws_credentials() == expected


def test_override_env_vars() -> None:
    kwargs = {"FOO": "foo", "BAR": None, "BAZ": "baz"}
    override_env_vars(**kwargs)
    for k, v in kwargs.items():
        if v is None:
            assert k not in os.environ
        else:
            assert os.environ.get(k) == v


@pytest.fixture(autouse=True)
def cleanup_env() -> Generator[None, None, None]:
    initial_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(initial_env)
