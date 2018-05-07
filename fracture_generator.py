"""
.. module:: fracture_update.py
   :synopsis: randomly generate fracture and update fractures
.. moduleauthor:: Shiyi Li

"""

from numpy import random
from string import Template

def fracture_update(self, fix_fractures):

    fix_centers = fix_fractures['centers']
    fix_normal_vectors = fix_fractures['normal_vectors']

    # Generate random radius for fixed fracture
    if radii_stats['Distribution'] == 'Normal':
        radii = random.normal(
            loc=radii_stats['mu'],
            scale=radii_stats['std'],
            size=[5, 1])
    elif radii_stats['Distribution'] == 'Uniform':
        radii = random.uniform(
            low=radii_stats['low'],
            high=radii_stats['high'],
            size=[5, 1])

    # Generate random aspect ratio for fixed fracture
    if aspect_ratio_stats['Distribution'] == 'Normal':
        aspect_ratio = random.normal(
            loc=aspect_ratio_stats['mu'],
            scale=aspect_ratio_stats['std'],
            size=[5, 1])
    elif aspect_ratio_stats['Distribution'] == 'Uniform':
        aspect_ratio = random.uniform(
            low=aspect_ratio_stats['low'],
            high=aspect_ratio_stats['high'],
            size=[5, 1])

    user_define_fracs = {'nUserEll': 5,
                         'Radii': radii,
                         'Aspect_Ratio': aspect_ratio,
                         'Translation': fix_centers,
                         'Normal': fix_normal_vectors}

    return user_define_fracs