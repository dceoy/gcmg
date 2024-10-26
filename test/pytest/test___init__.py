#!/usr/bin/env python

from importlib import reload

from pytest_mock import MockerFixture

import gcmg


def test_version_with_package(mocker: MockerFixture) -> None:
    package_version = "1.2.3"
    mocker.patch("importlib.metadata.version", return_value=package_version)
    mocker.patch("gcmg.__package__", new="gcmg")
    reload(gcmg)
    assert gcmg.__version__ == package_version
