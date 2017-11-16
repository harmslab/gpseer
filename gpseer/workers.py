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
            g = model.gpm.missing_genotypes
        elif genotypes == 'obs':
            g = model.gpm.genotypes
        else:
            g = model.gpm.complete_genotypes
    
    else:
        raise ValueError("genotypes must be 'missing', 'obs', or 'complete'.")
    
    # Predict using current model.
    p = model.predict(X=genotypes)
    
    # Return results as DataFrame
    data = dict(zip(g, p))
    predictions = pd.DataFrame(data, index=['max_likelihood'])
    return predictions

def run_pipeline(reference, gpm, model, genotypes='missing'):
    """Run fits and predictions in series without leaving current node."""
    # Run the ML fits
    model = setup(reference, gpm, model)
    model = run_fits(model)
    predictions = run_predictions(model, genotypes=genotypes)
    return model, predictions

def sample_fits(model, n_samples=10):
    """Use the BayesianSampler to possible models given the data."""
    # Initialize a Bayesian Sampler.
    sampler = BayesianSampler(model)
    
    # Sample models using Bayesian sampler and return
    samples, end_state = sampler.sample(n_samples, n_steps=n_samples)
    return samples, end_state

def sample_predictions(model, samples, bins, genotypes='missing'): 
    """Get the prediction posterior distributions from the samples.
    
    Parameters
    ----------
    model : 
        initialized epistasis model with a genotype-phenotype map.
    samples : 
        model parameters sampled by `sample_fits`
    bins : np.array
        Bins to use for histogram (must be an array)
    genotypes : str
        genotypes to predict.
        
    Returns
    -------
    predictions_df : pandas.DataFrame
        A dataframe with genotypes as columns, and bins as the index.
    """   
    if genotypes in ['missing', 'complete', 'obs']:

        if genotypes == 'missing':
            g = model.gpm.missing_genotypes
        elif genotypes == 'obs':
            g = model.gpm.genotypes
        else:
            g = model.gpm.complete_genotypes
    
    else:
        raise ValueError("genotypes must be 'missing', 'obs', or 'complete'.")
    
    # Initialize a predictions array. (rows are samples, and cols are genotypes)
    predictions = np.empty((len(samples), len(g)), dtype=float)
    
    # Predict and write.
    for i, sample in enumerate(data):
        # Get predictions
        p_sample = model.hypothesis(X=genotypes, thetas=samples[i])
        
        # Broadcast predictions into data array.
        predictions[:, i] = p_sample
        
    # histogram predictions array along the genotypes axis
    hists = np.histogram(predictions, axis=1)
    
    # Map genotype to their histogram
    data = dict(zip(g, hists))    
    predictions_df = pd.DataFrame(data=data, index=bins)
    return predictions_df
# 
# def sample_bayes_pipeline(reference, gpm, model, n_samples=10):
#     # Run worker pipeline on lone worker!
#     new_model = setup(reference, gpm, model)
#     new_model = fit(reference, new_model)
#     sampler = sample(reference, new_model, n_samples=n_samples)
#     sampler = predict(reference, sampler)
