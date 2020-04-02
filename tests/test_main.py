
import pytest

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.script_launch_mode('subprocess')


def test_help(script_runner, capsys):
    script_runner.run('gpseer', 'estimate-ml')