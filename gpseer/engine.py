

class Engine(object):
    
    def _setup_worker(self, key, gpm, model):
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

        # Store model
        return new_model

    def _fit_worker(self, key, model):



    def _run_worker(self, key, gpm, model):
        self._setup_worker()
