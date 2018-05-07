"""
.. module:: distribution.py
   :synopsis: A collection of common probability distributions for stochastic
.. moduleauthor:: Shiyi Li

"""

import numpy as np
from numpy.random import normal

class Normal_distribution:

    def __init__(self, mu=0, std=1, **kwargs):
        self.mu = mu
        self.std = std

    def prob_density(self, x):
        return 1/np.sqrt(2*np.pi*self.std**2) * np.exp(-(x-self.mu)**2/(2*self.std**2))

    def make_proposal(self, current_state):
        return normal(loc= current_state, scale=self.std)

