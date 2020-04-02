import pytest
import shutil


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

