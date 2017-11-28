__doc__ = "Factory for GPSeer objects."""

from .utils import EngineError

from . import multiple
from . import single

def GPSeer(gpm, model, bins,
    genotypes='missing',
    sample_weight=None, 
    client=None,
    single_reference=True, 
    db_path="database/"):
    """Creates a sampling engine."""
    
    # Tell whether to serialize or not.
    if client is None: 
        if single_reference:
            engine = single.SerialEngine
        else:
            engine = multiple.SerialEngine
        cls = engine(gpm=gpm, 
            model=model, 
            bins=bins, 
            genotypes=genotypes,
            sample_weight=sample_weight, 
            db_path=db_path)
    else:
        if single_reference:
            engine = single.DistributedEngine
        else:
            engine = multiple.DistributedEngine
        cls = engine(client, 
            gpm=gpm, 
            model=model, 
            bins=bins, 
            genotypes=genotypes,
            sample_weight=sample_weight, 
            db_path=db_path)
        #except:
        #    raise EngineError('client argument is invalid. Must be "serial" or "distributed".')

    # Initialize the engine.
    return cls
