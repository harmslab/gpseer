import os
import pytest

from epistasis.models import EpistasisLinearRegression

from . import gpm, model, clean_database
from ..distributed import DistributedEngine


from dask.distributed import Client

@pytest.fixture
def client():
    c = Client()
    return c


@pytest.mark.usefixtures("clean_database")
class TestEngine(object):
    
    def test_setup(self, client, gpm, model):        
        engine = DistributedEngine(client, gpm, model)
        engine.setup()
        
        # Check that a model map is added to the object
        assert hasattr(engine, "model_map")
        
        # Check one of the items in the model_map
        example_item =  list(engine.model_map.values())[0]
        assert 'model' in example_item
        assert isinstance(example_item['model'], EpistasisLinearRegression)

    def test_fit(self, client, gpm, model):
        engine = DistributedEngine(client, gpm, model)
        engine.setup()
        engine.fit()
        
        model = engine.model_map['00']['model']
        assert hasattr(model, 'coef_')
        assert len(model.coef_) > 0
        s
    def test_sample(self, client, gpm, model):
        engine = DistributedEngine(client, gpm, model)
        engine.setup()
        engine.fit()
        engine.sample(n_samples=5)
        
        # Check sampler was added to mapping system
        mapping_item = engine.model_map['00']
        assert 'sampler' in mapping_item
        
        # Assert
        sampler = mapping_item['sampler']
        assert hasattr(sampler, 'samples')

    def test_predict(self, client, gpm, model):
        engine = DistributedEngine(client, gpm, model)
        engine.setup()
        engine.fit
        engine.sample(n_samples=5)
        engine.predict()
        
         
        
    def test_run(self, client, gpm, model):
        engine = DistributedEngine(client, gpm, model)
        engine.run(n_samples=5)