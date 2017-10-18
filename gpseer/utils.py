class EngineError(Exception):
    """Rename exception for problems with distributed client."""

class SubclassError(Exception):
    """Exception raised when a method is called in a parent class that
    only works in a child class."""
