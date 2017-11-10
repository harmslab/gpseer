import os
import warnings
import numpy as np
import pandas as pd

from gpmap import GenotypePhenotypeMap
from epistasis.sampling.bayesian import BayesianSampler

def setup(reference, gpm, model):
    """Initializes an epistasis model and genotype-phenotype map object.
    
    Parameters
    ----------
    reference : 
    gpm : 
    model :
    
    Returns
    -------
    new_model :
    
    """
    # Build a GenotypePhenotypeMap with a new reference state.
    new_gpm = GenotypePhenotypeMap(reference, # New reference state.
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

def run_fits(model):
    """Estimate the most likely solution to the data given the model."""
    # Ignore warnings.
    warnings.simplefilter('ignore', RuntimeWarning)
    
    # Fit a model
    model.fit()
    return model

def run_predictions(model, genotypes='missing'):
    """Predict missing phenotypes given the most likely models.
    
    Parameters
    ----------
    model : 
        a model from the `epistasis.models` module.
    genotypes : str
        what group of genotypes should I predict? Options are 'missing', 'obs', or 'complete'.

    Returns
    -------
    predictions : pandas.DataFrame
        two column DataFrame with 'genotypes' and 'phenotypes' columns. 
    """
    if genotypes in ['missing', 'complete', 'obs']:
    
        if genotypes == 'missing':
            data = {'genotypes':model.gpm.missing_genotypes}
        elif genotypes == 'obs':
            data = {'genotypes':model.gpm.genotypes}
        else:
            data = {'genotypes':model.gpm.complete_genotypes}
    
    else:
        raise ValueError("genotypes must be 'missing', 'obs', or 'complete'.")
    
    # Predict using current model.
    data['phenotypes'] = model.predict(X=genotypes)
    
    # Return results as DataFrame
    predictions = pd.DataFrame(data)
    return predictions

def run_maximum_likelihood(model, genotypes='missing'):
    """Run fits and predictions in series without leaving current node."""
    # Run the ML fits
    model = run_fits(model)
    
    # Make predictions based on fits
    predictions = run_predictions(model)
    return model, predictions

def sample_model(model, n_samples=10):
    """Use the BayesianSampler to possible models given the data."""
    # Initialize a Bayesian Sampler.
    sampler = BayesianSampler(model)
    
    # Sample models using Bayesian sampler and return
    sampler.sample(n_samples)
    
    
    return sampler
    
def sample_predictions(sampler, X='missing'): 
    """Use the sampled models to predict phenotypes."""   
    # Predict and write.
    sampler.predict(X=X)
    return sampler

def run_bayesian(reference, gpm, model, n_samples=10):
    # Run worker pipeline on lone worker!
    new_model = setup(reference, gpm, model)
    new_model = fit(reference, new_model)
    sampler = sample(reference, new_model, n_samples=n_samples)
    sampler = predict(reference, sampler)
