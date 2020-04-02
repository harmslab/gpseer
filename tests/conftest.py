import pytest


@pytest.fixture
def test_example_dir(tmp_path):
    example = tmp_path.joinpath('test_example')
    example.mkdir()
    return example

@pytest.fixture
def
