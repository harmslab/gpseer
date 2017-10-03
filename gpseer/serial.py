import pandas as pd

from epistasis.sampling.bayesian import BayesianSampler

class SerialEngine(object):
    """"""
    @staticmethod
    def setup(gpm, model):        
        # keys
        keys = gpm.complete_genotypes
        
        # Set up models. Copies the GenotypePhenotypeMap
        model_map = {}
        for key in keys:
            # Build a GenotypePhenotypeMap with a new reference state.
            new_gpm = GenotypePhenotypeMap( key, # New reference state.
                gpm.genotypes,
                gpm.phenotypes,
                stdeviations=gpm.stdeviations,
                n_replicates=gpm.n_replicates,
                mutations=gpm.mutations) 
            
            # initialize a completely new model object with same parameters
            new_model = model.__class__(**model.model_specs)
    
            # Add genotype-phenotype map.
            new_model.add_gpm(new_gpm)
            new_model.fit()

            # Store model
            items = dict(model=new_model, sampler=BayesianSampler(new_model))
            model_map[key] = items
    
    @staticmethod
    def fit(model_map):
        """"""
        for key, items in model_map.items():
            model = items["model"]
            model.fit()

    @staticmethod            
    def sample(model_map):
        """"""
        for key, items in model_map.items():
            sampler = items["sampler"]
            sampler.sample(10)

    @staticmethod        
    def predict(model_map):
        """"""
        dfs = []
        for key, items in model_map.items():
            sampler = items["sampler"]
            sampler.predict()
            dfs.append(sampler.predictions)
        # Need to do something with data    
        data = pd.concat(dfs)
    
    @staticmethod
    def collect(model_map):
        """"""
        pass
