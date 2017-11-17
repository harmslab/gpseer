__doc__ = "Factory for GPSeer objects."""

from .engine import EngineError
from .serial import SerialEngine
from .distributed import DistributedEngine

def GPSeer(gpm, model, bins, client=None, db_path="database/"):
    """Creates a sampling engine.
    
    
    """
    # Tell whether to serialize or not.
    if client == None: 
        cls = SerialEngine(gpm, model, bins, db_path=db_path)
    elif client != None:
        cls = DistributedEngine(client, gpm=gpm, model=model, bins=bins, db_path=db_path)
    else:
        raise EngineError('client argument is invalid. Must be "serial" or "distributed".')
    return cls
