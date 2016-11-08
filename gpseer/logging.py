import os
import sys
import time
from functools import wraps

def log(Predictor, method, *args, **kwargs):
    @wraps(method)
    def prints_to_file(*args, **kwargs):
        """
        """
        # Get the method's name
        name = method.__name__
        # Write to file
        return method(*args, **kwargs)
    return prints_to_file


class Logger(object):

    def __init__(Predictor):
        """
        """

    def log(self):
        """
        """
