__doc__ = "Factory for GPSeer objects."""

from .utils import EngineError

import .multiple
import .single

def GPSeer(gpm, model, bins, sample_weights=None, client=None, single_reference=True, db_path="database/"):
    """Creates a sampling engine."""
    # Tell whether to serialize or not.
    if client != None: 
        if single_reference:
            cls = single.DistributedEngine(client, gpm=gpm, model=model, bins=bins, sample_weights=sample_weights, db_path=db_path)
        else:
            cls = multiple.DistributedEngine(client, gpm=gpm, model=model, bins=bins, sample_weights=sample_weights, db_path=db_path)
    else:
        raise EngineError('client argument is invalid. Must be "serial" or "distributed".')
    return cls
