
import pytest

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.script_launch_mode('subprocess')


def test_estimate_ml(script_runner):
    script_runner.run('gpseer', 'estimate-ml')


def test_goodness_of_fit(script_runner):
    script_runner.run('gpseer', 'goodness-of-fit')