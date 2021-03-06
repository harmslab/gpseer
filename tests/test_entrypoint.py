
import pytest

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.script_launch_mode('subprocess')

# Use the following tests to verify that the console entrypoints are correct.
def test_estimate_ml_help(script_runner):
    ret = script_runner.run('gpseer', 'estimate-ml', '-h')
    assert ret.success


def test_cross_validate_help(script_runner):
    ret = script_runner.run('gpseer', 'cross-validate', '-h')
    assert ret.success


