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


@pytest.fixture
def example_dir(tmp_path):
    example = tmp_path.joinpath('test_example')
    example.mkdir()
    return example


@pytest.fixture
def example_input_file(example_dir):
    # Copy files to example_test_dir
    input_file = example_dir.joinpath('input_file')
    shutil.copyfile(
        src='examples/example-train.csv',
        dst=str(input_file)
    )
    return input_file


