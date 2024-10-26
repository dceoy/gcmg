#!/usr/bin/env python

from importlib.metadata import version

__version__ = version(__package__) if __package__ else None
