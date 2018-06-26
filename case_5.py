from dfninverse import DFNINVERSE
from sampler import MCMCSampler
import subprocess
import sys, getopt, os
import numpy as np


if __name__ == '__main__':

    ncpu = 6

    case_dir = '/cluster/scratch/lishi/Case5_2x2x2_n'

    domainSize = [2.0, 2.0, 2.0]

    stations = [(-0.6, -0.16, 0), (0, 0, -0.4), (0, 0, 0), (0, 0, -0.8)
                # (0.8, 0.8, 0.4), (0.8, 0.8, -0.8), (0.8, -0.8, 0.4),(0.8, -0.8, -0.8),
               ]

    dfn_synthetic = np.asarray(
        [[-0.8, 0, 0, 0, np.pi / 2, 1.6],
         [0.6, 0, 0.4, 0, np.pi / 2, 1.6],
         [0.8, 0, -0.8, 0, np.pi / 2, 1.6],
         [0.2, 0, 0.4, 0, 0, 1.6],
         [-0.2, 0, -0.4, 0, 0, 0.8],
         [0, 0, -0.4, 0, np.pi/2, 0.5]])

    dfn_initial = np.asarray(
        [[-0.8, 0, 0, 0, np.pi / 2, 1.6],
         [0.6, 0, 0.4, 0, np.pi / 2, 1.6],
         [0.8, 0, -0.8, 0, np.pi / 2, 1.6],
         [0, 0, -0.4, 0, np.pi / 2, 0.5],
         [0.2, 0, 0.4, 0, 0, 1.6]])

    sig_f = np.asarray([0, 0, 0, 1, 1])
    sig_v = np.asarray([0.1, 0, 0, 0, 0, 0])

    low_bound = [0, -domainSize[0]/2, -domainSize[1]/2, -domainSize[2]/2, 0, 0, 0]
    high_bound = [10, domainSize[0]/2, domainSize[1]/2, domainSize[2]/2, np.pi, np.pi, 10]

    max_iter = 1000

    dodr = 0

    mv = ['B']#, 'B', 'K']

    # define flow condition: [inflow_boundary, inflow_pressure, outflow_bounday, outflow_pressure]
    # Unit: MPa
    flow_condition = ['front', 5, 'right', 1]

    reference_dfn = np.asarray([0, 0, -0.4, 0, 0, 0.8])
    # Run synthetic model

    synthetic_job = case_dir + '/synthetic'

    syn_engine = DFNINVERSE(synthetic_job, stations, domainSize, flow_condition, ncpu)
    syn_data = syn_engine.run_forward(dfn_synthetic)
    # print(syn_data)
    syn_engine.gen_3d_obs_points_plot(stations, radius=np.average(domainSize) / 50)

    # Run inversion

    inverse_job = case_dir + '/inverse'

    dfninv = DFNINVERSE(inverse_job, stations, domainSize, flow_condition, ncpu)

    sp = MCMCSampler(dfninv, syn_data,
                     var_sigma=[sig_f, sig_v], moves=mv,
                     prior_range=[low_bound, high_bound],
                     reference=reference_dfn)

    chain = sp.sample(dfn_initial, max_iter, dodr)









