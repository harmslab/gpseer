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

def run_fits(model, sample_weights=None):
    """Estimate the most likely solution to the data given the model."""
    # Ignore warnings.
    warnings.simplefilter('ignore', RuntimeWarning)
    
    # Fit a model
    model.fit(sample_weights=sample_weights)
    return model

def run_predictions(model, genotypes='missing'):
    """Predict missing phenotypes given the most likely models.
    
    Parameters
    ----------
    model : 
        a model from the `epistasis.models` module.
    genotypes : array
        what group of genotypes should I predict? Options are 'missing', 'obs', or 'complete'.
    
    Returns
    -------
    predictions : pandas.DataFrame
        two column DataFrame with 'genotypes' and 'phenotypes' columns. 
    """
    # Store the predicted genotypes
    if genotypes in ['missing', 'complete', 'obs']:
        if genotypes == 'missing':
            gs = model.gpm.missing_genotypes
        elif genotypes == 'obs':
            gs = model.gpm.genotypes
        else:
            gs = model.gpm.complete_genotypes
    else:
        raise ValueError("genotypes must be 'missing', 'obs', or 'complete'.")

    # Predict using current model.
    ps = model.predict(X=genotypes)
    
    # Return results as DataFrame
    data = dict(zip(gs, ps))
    predictions = pd.DataFrame(data, index=['max_likelihood'])
    return predictions

def run_pipeline(reference, gpm, model, sample_weights=None, genotypes='missing'):
    """Run fits and predictions in series without leaving current node."""
    # Run the ML fits
    model = setup(reference, gpm, model)
    model = run_fits(model, sample_weights=sample_weights)
    predictions = run_predictions(model, genotypes=genotypes)
    return model, predictions

def sample_fits(model, n_samples=10, previous_state=None):
    """Use the BayesianSampler to possible models given the data.
    
    Parameters
    ----------
    model : 
        initialized epistasis model with a genotype-phenotype map.
    n_samples : int
        number of steps to take in sampler.
    previous_state : dict
        dictionary with data from latest step in MCMC walk. Must have 'rstate', 'lnprob',
        and 'pos' as keys.
        
    Returns
    -------
    samples : ndarray
        Samples from MCMC walk for all model parameters. shape = (n_samples, number of coefs) 
    end_state : dict
        dictionary describing the final step in the MCMC walk. Has keys 'rstate',
        'lnprob', and 'pos' as keys.
    """
    # Initialize a Bayesian Sampler.
    sampler = BayesianSampler(model)
    
    # Sample models using Bayesian sampler and return
    samples, end_state = sampler.sample(n_steps=n_samples, previous_state=previous_state)
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
    genotypes : list
        genotypes to predict.
        
    Returns
    -------
    predictions_df : pandas.DataFrame
        A dataframe with genotypes as columns, and bins as the index.
    """
    # Store the predicted genotypes
    if genotypes in ['missing', 'complete', 'obs']:
        if genotypes == 'missing':
            gs = model.gpm.missing_genotypes
        elif genotypes == 'obs':
            gs = model.gpm.genotypes
        else:
            gs = model.gpm.complete_genotypes
    else:
        raise ValueError("genotypes must be 'missing', 'obs', or 'complete'.")    
    
    # Initialize a predictions array. (rows are samples, and cols are genotypes)
    predictions = np.empty((len(samples), len(gs)), dtype=float)
    
    # Predict and write.
    for i, sample in enumerate(samples):
        # Get predictions
        p_sample = model.hypothesis(X=genotypes, thetas=sample)
        
        # Broadcast predictions into data array.
        predictions[i, :] = p_sample
        
    # histogram predictions array along the genotypes axis
    data = {g : np.histogram(predictions[:, i], bins=bins)[0] for i, g in enumerate(gs)}
    
    # Return a dataframe.
    predictions_df = pd.DataFrame(data=data, index=bins[1:])
    return predictions_df

def sample_pipeline(reference, gpm, model, n_samples, bins, genotypes='missing', 
    previous_state=None):
    """Sample an epistasis model."""
    # Run ML pipeline to get ML solutions.
    model, ml_predictions = run_pipeline(reference, gpm, model)
    
    # Sample model parameters
    samples, end_state = sample_fits(model, 
        n_samples=n_samples, 
        previous_state=previous_state)
        
    # Use the samples to predict phenotypes.
    predictions = sample_predictions(model, samples, bins)
    return model, end_state, ml_predictions, predictions
# 
# def sample_pipeline(model, n_samples, previous_state):
#     """"""
#     # Sample model parameters
#     samples, end_state = sample_fits(model, 
#         n_samples=n_samples, 
#         previous_state=previous_state)
#         
#     # Use the samples to predict phenotypes.
#     predictions = sample_predictions(model, samples, bins)
#     return end_state, predictions
