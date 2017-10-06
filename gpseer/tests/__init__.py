import os
import shutil
import pytest

from gpmap import GenotypePhenotypeMap
from epistasis.models import EpistasisLinearRegression

@pytest.fixture()
def gpm():
    """Build a genotype-phenotype map."""
    wildtype = "00"
    genotypes = ["00", "01", "10", "11"]
    phenotypes = [1, 1.5, 1.9, 4.0]
    stdeviations = [0.05, 0.05, 0.05, 0.05]
    mutations = {0:["0","1"],1:["0","1"]}
    return GenotypePhenotypeMap(
        wildtype,
        genotypes,
        phenotypes,
        stdeviations=stdeviations,
        mutations=mutations)
    
@pytest.fixture()
def model():
    """"""
    return EpistasisLinearRegression(order=2, model_type="local")

@pytest.fixture()
def clean_database():
    yield clean_database
    shutil.rmtree("database/")
