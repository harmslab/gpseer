import numpy as np
import matplotlib.pyplot as plt

class GenotypePlotting(object):
    """
    """
    def __init__(self, genotype):
        self.genotype = genotype

    def distribution(self, **kwargs):
        plt.bar(self.genotype.histogram())
        
