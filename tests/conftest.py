import io
import logging
import shutil

import pytest


from gpseer.main import setup_logger

@pytest.fixture
def console_log():
    """An io stream with the console's log."""
    stream = io.StringIO()
    return stream


@pytest.fixture
def logger(console_log):
    return setup_logger(console_log)

