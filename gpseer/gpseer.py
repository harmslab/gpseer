from .engine import EngineError
from .serial import SerialEngine
from .distributed import DistributedEngine

def GPSeer(gpm, model, client=None, db_path="database/"):
    """Factory for sampling engines."""
    # Tell whether to serialize or not.
    if client == None: 
        cls = SerialEngine(gpm, model, db_path=db_path)
    elif client != None:
        cls = DistributedEngine(gpm, model, client=client, db_path=db_path)
    else:
        raise EngineError('client argument is invalid. Must be "serial" or "distributed".')
    return cls
