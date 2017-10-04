import os
import warnings
import numpy as np
import pandas as pd

from gpmap import GenotypePhenotypeMap
from epistasis.sampling.bayesian import BayesianSampler

def setup(reference, gpm, model):
    # Build a GenotypePhenotypeMap with a new reference state.
    new_gpm = GenotypePhenotypeMap( reference, # New reference state.
        gpm.genotypes,
        gpm.phenotypes,
        stdeviations=gpm.stdeviations,
        n_replicates=gpm.n_replicates,
        mutations=gpm.mutations) 
    
    # initialize a completely new model object with same parameters
    new_model = model.__class__(**model.model_specs)

    # Add genotype-phenotype map.
    new_model.add_gpm(new_gpm)

    # Store model
    return new_model

def fit(reference, model):
    # Ignore warnings.
    warnings.simplefilter('ignore', RuntimeWarning)
    # Fit a model
    model.fit()
    return model

def sample(reference, model, n_samples=10):
    # Attach Bayesian Sampler and sample!
    sampler = BayesianSampler(model)
    sampler.sample(n_samples)
    return sampler
    
def predict(reference, sampler, db_path="database/"):
    # Path to write out predictions to disk
    path = os.path.join(db_path, "{}.csv".format(reference))
    
    # Predict and write.
    sampler.predict()
    sampler.predictions.to_csv(path)
    return sampler

def run(reference, gpm, model, n_samples=10, db_path="database/"):
    # Run worker pipeline on lone worker!
    new_model = setup(reference, gpm, model)
    new_model = fit(reference, new_model)
    sampler = sample(reference, new_model, n_samples=n_samples)
    sampler = predict(reference, sampler, db_path=db_path)
