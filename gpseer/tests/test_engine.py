import os
import pytest
from . import gpm, model, clean_database
from ..engine import Engine

@pytest.mark.usefixtures("clean_database")
class TestEngine(object):
    
    def test_init(self, gpm, model):
        engine = Engine(gpm, model)
        assert hasattr(engine, "gpm")
        assert hasattr(engine, "model")
        assert hasattr(engine, "db_path")
        assert os.path.exists("database")
