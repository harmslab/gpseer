import os
import glob
import numpy as np
import pandas as pd
from functools import wraps
from collections import Counter
from gpmap.utils import hamming_distance

from .. import workers
from ..engine import Engine, save_engine

# Import Dask stuff for distributed computing!
from dask import delayed, compute


class DistributedEngine(Engine):
    """GPSeer engine that distributes the work across all resources using Dask.
    """
    @wraps(Engine)
    @save_engine
    def __init__(self, client=None, *args, **kwargs):
        # Set up Engine
        super(DistributedEngine, self).__init__(*args, **kwargs)

        # Reference client for distributed computing
        self.client = client
        self.keys = sum(self.client.ncores().values())
        self.perspective = 'single'

        # Internal storage items.
        self.map_of_mcmc_states = {i: None for i in range(self.keys)}
        self.map_of_models = {i: None for i in range(self.keys)}
        self.map_of_predictions = {i: None for i in range(self.keys)}

        # Prepare storage in memory for prediction histograms using a DataFrame
        self.map_of_sampled_predictions = {}
        for i in range(self.keys):
            # Fill that dataframe with zeros.
            df = pd.DataFrame(0, index=self.bins[1:],
                              columns=self.predicted_genotypes)

            # Store that dataframe in a dictionary.
            self.map_of_sampled_predictions[i] = df

    @wraps(Engine.setup)
    @save_engine
    def setup(self):
        # Distribute the work using Dask.
        processes = []
        for i in range(self.keys):
            # Build process for this model.
            process = delayed(workers.setup)(self.gpm.wildtype, self.gpm,
                                             self.model)

            # Add process to list of processes
            processes.append(process)

        # Compute processes on distributed network.
        results = compute(*processes, get=self.client.get)

        # Organize the results
        for i in range(self.keys):
            # Get model from results distributed
            new_model = results[i]

            # Store model
            self.map_of_models[i] = new_model

    @wraps(Engine.run_fits)
    @save_engine
    def run_fits(self):
        # Distribute the work using Dask.
        processes = []
        for i in range(self.keys):
            # Build process for this model.
            process = delayed(workers.run_fits)(
                self.map_of_models[i], sample_weight=self.sample_weight)

            # Add process to list of processes
            processes.append(process)

        # Compute processes on distributed network.
        results = compute(*processes, get=self.client.get)

        # Store the results
        for i in range(self.keys):
            self.map_of_models[i] = results[i]

    @wraps(Engine.run_predictions)
    @save_engine
    def run_predictions(self):
        # Distribute the work using Dask.
        processes = []
        for i in range(self.keys):
            # Build process for this model.
            g = genotypes
            process = delayed(workers.run_predictions)(self.map_of_models[i],
                                                       genotypes=g)

            # Add process to list of processes
            processes.append(process)

        # Compute processes on distributed network.
        results = compute(*processes, get=self.client.get)

        # Zip predictions
        for i in range(self.keys):
            self.map_of_predictions[i] = results[i]

    @wraps(Engine.run_pipeline)
    @save_engine
    def run_pipeline(self):
        # Distribute the work using Dask.
        processes = []
        for i in range(self.keys):
            # Build process for this model.
            process = delayed(workers.run_pipeline)(self.gpm.wildtype,
                                                    self.gpm,
                                                    self.model,
                                                    genotypes=self.genotypes)

            # Add process to list of processes
            processes.append(process)

        # Compute processes on distributed network.
        results = compute(*processes, get=self.client.get)

        # Collect results
        for i in range(self.keys):
            self.map_of_models[i] = results[i][0]
            self.map_of_predictions[i] = results[i][1]

    @wraps(Engine.sample_fits)
    def sample_fits(self, n_samples, n_burn=0):
        # Proper order check
        if hasattr(self, 'map_of_models') is False:
            raise Exception('Try running `run_fits` '
                            'before running this method.')

        # Distribute the work using Dask.
        processes = []
        for i in range(self.keys):
            # Build process for this model.
            state = self.map_of_mcmc_states[i]
            process = delayed(workers.sample_fits)(self.map_of_models[i],
                                                   n_samples=n_samples,
                                                   n_burn=n_burn,
                                                   previous_state=state)

            # Add process to list of processes
            processes.append(process)

        # Compute processes on distributed network.
        results = compute(*processes, get=self.client.get)

        # Store mcmc states.
        self.map_of_mcmc_states = {i: results[i][1] for i in
                                   range(self.keys)}

        # Map of samples to return
        return {i: results[i][0] for i in range(self.keys)}

    @wraps(Engine.sample_predictions)
    def sample_predictions(self, map_of_model_samples):
        # Distribute the work using Dask.
        processes = []
        for i in range(self.keys):
            process = delayed(workers.sample_predictions)(
                self.map_of_models[i],
                map_of_model_samples[i],
                self.bins, genotypes=self.genotypes)

            # Add process to list of processes
            processes.append(process)

        # Compute processes on distributed network.
        results = compute(*processes, get=self.client.get)

        # Organize the results.
        return {i: results[i] for i in range(self.keys)}

    @wraps(Engine.sample_pipeline)
    @save_engine
    def sample_pipeline(self, n_samples, n_burn=0):
        # Check that the ML pipeline has been run.
        if None is list(self.map_of_predictions.values())[0]:
            self.run_pipeline()

        # Distribute the work using Dask.
        processes = []
        gs = self.genotypes
        for i in range(self.keys):
            model = self.map_of_models[i]
            state = self.map_of_mcmc_states[i]
            process = delayed(workers.sample_pipeline)(model,
                                                       n_samples, self.bins,
                                                       n_burn=n_burn,
                                                       genotypes=gs,
                                                       previous_state=state)

            # Add process to list of processes
            processes.append(process)

        # Compute processes on distributed network.
        results = compute(*processes, get=self.client.get)

        # Parse results from workers.
        for i in range(self.keys):
            # Store end state of MCMC walk.
            self.map_of_mcmc_states[i] = results[i][0]

            # Add to new values to predictions dataframe.
            df = self.map_of_sampled_predictions[i]
            df = df.add(results[i][1], fill_value=0)

            # Store dataframe in memory.
            self.map_of_sampled_predictions[i] = df

    @property
    def ml_results(self):
        """Get the maximum likelihood results"""
        # Get example predictions DataFrame
        data = {}
        for genotype in self.predicted_genotypes:
            for i in range(self.keys):
                # Get max_likelihood
                val = self.map_of_predictions[i][genotype]['max_likelihood']
                data[genotype] = [val]

        df = pd.DataFrame(data, index=['max_likelihood'])
        return df

    @property
    def results(self):
        """Get dataframe of prediction results."""
        df = self.ml_results

        # Add histograms
        data = {g: [] for g in self.predicted_genotypes}
        if hasattr(self, 'map_of_sampled_predictions'):
            # Get histograms
            mapping = self.map_of_sampled_predictions
            for genotype in self.predicted_genotypes:
                arr = np.zeros(len(self.bins) - 1)

                # Construct histograms
                for i in range(self.keys):
                    arr += np.array(mapping[i][genotype].values)  # * priors[i]
                data[genotype] += list(arr)

            # Append posterior distributions to dataframe
            df2 = pd.DataFrame(data, index=list(self.bins[1:]))
            df = df.append(df2)

        return df
