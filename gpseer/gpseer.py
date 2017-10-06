from .engine import EngineError
from .serial import SerialEngine
from .distributed import DistributedEngine

def GPSeer(gpm, model, engine="serial", db_path="database/"):
    """Factory for sampling engines."""
    # Tell whether to serialize or not.
    if engine == "serial": 
        cls = SerialEngine(gpm, model, db_path=db_path)
    elif engine == "distributed":
        cls = DistributedEngine(gpm, model, db_path=db_path)
    else:
        raise EngineError('client argument is invalid. Must be "serial" or "distributed".')
    return cls
