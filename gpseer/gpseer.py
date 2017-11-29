__doc__ = "Factory for GPSeer objects."""

from .utils import EngineError

from . import multiple
from . import single


def GPSeer(gpm, model, bins, genotypes='missing', sample_weight=None,
           client=None, perspective='single', db_dir="database/"):
    """Main entry point to the GPSeer package. This function is a factory for
    creating a engine for infering missing data in a sparsely sampled
    genotype-phenotype map.

    The returned engine will have methods for estimating both the Maximum
    likelihood predictions for all missinge genotype-phenotypes and for
    sampling their Bayesian, posterior distributions.

    Parameters
    ----------
    gpm : GenotypePhenotypeMap object
        Genotype-phenotype data stored in a GenotypePhenotypeMap object.
    model : epistasis.models object
        The epistasis model to use.
    bins : np.array
        Histogram bins for output posterior distributions.
    genotypes : str (default='missing')
        What genotypes do you want to predict? 'missing' or 'complete'.
    sample_weight : str (default=None)
        If 'relative', will use a relative err as the objective function
        minimize.
    client : dask.distributed.Client (default=None)
        A Dask Client for distributed computing.
    perspective: str (default='single')
        Sample a single model from a singel reference genotype (wildtype) or
        use a multiple reference state model.
    db_dir : str (default='database')
        Directory to store progress.
    """
    # Tell whether to serialize or not.
    if client is None:
        if perspective == 'single':
            engine = single.SerialEngine
        else:
            engine = multiple.SerialEngine
        cls = engine(gpm=gpm,
                     model=model,
                     bins=bins,
                     genotypes=genotypes,
                     sample_weight=sample_weight,
                     db_dir=db_dir)
    else:
        if perspective == 'single':
            engine = single.DistributedEngine
        else:
            engine = multiple.DistributedEngine
        cls = engine(client,
                     gpm=gpm,
                     model=model,
                     bins=bins,
                     genotypes=genotypes,
                     sample_weight=sample_weight,
                     db_dir=db_dir)

    # Initialize the engine.
    return cls
