from dfninverse import DFNINVERSE
from sampler import MCMCSampler
import subprocess
import sys, getopt, os
import numpy as np


if __name__ == '__main__':

    ncpu = 4

    case_dir = '/cluster/scratch/lishi/Case2_1x1x1_cxyz'

    domainSize = [1.0, 1.0, 1.0]

    stations = [(0.4, 0.4, 0.2), (0.4, 0.4, -0.2), (0.4, -0.4, 0.2),
                (0.4, -0.4, -0.2), (-0.15, -0.08, 0.2), (-0.15, -0.08, 0)]

    dfn_synthetic = np.asarray(
        [[-0.4, 0, 0, 0, np.pi / 2, 0.8],
         [0.3, 0, 0.2, 0, np.pi / 2, 0.8],
         [0.4, 0, -0.2, 0, np.pi / 2, 0.8],
         [0.1, 0, 0.2, 0, 0, 0.8]])

    dfn_initial = np.asarray(
        [[-0.4, 0, 0, 0, np.pi / 2, 0.8],
         [0.3, 0, 0.2, 0, np.pi / 2, 0.8],
         [0.4, 0, -0.2, 0, np.pi / 2, 0.8],
         [-0.2, 0.2, 0, 0, 0, 0.8]])

    sig_f = np.asarray([0, 0, 0, 1])
    sig_v = np.asarray([0.1, 0.1, 0.1, 0, 0, 0])

    low_bound = [0, -domainSize[0]/2, -domainSize[1]/2, -domainSize[2]/2, 0, 0, 0]
    high_bound = [10, domainSize[0]/2, domainSize[1]/2, domainSize[2]/2, np.pi, np.pi, 10]

    max_iter = 500

    dodr = 0

    mv = 'D'

    # Run synthetic model

    synthetic_job = case_dir + '/synthetic'

    syn_engine = DFNINVERSE(synthetic_job, stations, domainSize, ncpu)
    syn_data = syn_engine.run_forward(dfn_synthetic)
    syn_engine.gen_3d_obs_points_plot(stations, radius=np.average(domainSize)/50)

    # Run inversion

    inverse_job = case_dir + '/inverse'

    dfninv = DFNINVERSE(inverse_job, stations, domainSize, ncpu)

    sp = MCMCSampler(dfninv, syn_data,
                     var_sigma=[sig_f, sig_v], moves=mv, prior_range=[low_bound, high_bound])

    chain = sp.sample(dfn_initial, max_iter, dodr)









