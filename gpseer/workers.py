import os
import warnings
import numpy as np
import pandas as pd

from gpmap import GenotypePhenotypeMap
from epistasis.sampling.bayesian import BayesianSampler

def setup(reference, gpm, model):
    """Initializes an epistasis model and genotype-phenotype map object."""
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
    """Fit the model."""
    # Ignore warnings.
    warnings.simplefilter('ignore', RuntimeWarning)
    # Fit a model
    model.fit()
    return model

def sample_model(reference, model, n_samples=10):
    """Use the BayesianSampler to possible models given the data."""
    # Attach Bayesian Sampler and sample!
    sampler = BayesianSampler(model)
    sampler.sample(n_samples)
    return sampler
    
def sample_predictions(reference, sampler): 
    """Use the sampled models to predict phenotypes."""   
    # Predict and write.
    sampler.predict()
    return sampler

def run(reference, gpm, model, n_samples=10):
    # Run worker pipeline on lone worker!
    new_model = setup(reference, gpm, model)
    new_model = fit(reference, new_model)
    sampler = sample(reference, new_model, n_samples=n_samples)
    sampler = predict(reference, sampler)
