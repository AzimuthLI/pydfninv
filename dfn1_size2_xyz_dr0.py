from dfninverse import DFNINVERSE
from sampler import MCMCSampler
from numpy.random import normal
import numpy as np
import pickle, os
from helper import stdout_redirect

if __name__ == '__main__':

    ncpu = 5

    case_dir = '/cluster/scratch/lishi/dfn1_size2_xyz_dr0'

    max_iter = 2000

    dodr = 0

    domainSize = [2.0, 2.0, 2.0]

    stations = [(0, -0.4, -0.8),  (0, 0.4, -0.8), (0.4, 0.4, 0.4), (0.4, -0.2, 0.5), (-0.6, 0, -0.46),
                (-0.2, -0.4, 0.74), (0, -0.16, -0.12), (0.32, 0.4, -0.16), (-0.8, 0, 0)]

    # Setup dfn : | x | y | z | phi | psi | radius | aspect_ratio | beta | n_vertices |
    dfn_synthetic = np.asarray(
        [[-0.8, 0.4, 0, 0, np.pi/2.2, 1.6, 1, 30, 5],
         [0.4, 0, 0.4, 0, np.pi/3, 1.2, 0.8, 0, 5],
         [0, 0, -0.8, 0, np.pi/3, 1.2, 1, 0, 5],
         [-0.4, 0, -0.96, 0, 0, 1.2, 0.7, 0, 5],
         [0, 0, 0.4, np.pi/6, np.pi*0.1, 1, 0.6, 0, 5]])
        # [[-1, 0, 0, 0, np.pi / 2, 3],[1.5, 0, 1, 0, np.pi / 2, 3],
        #  [1, 0, -1, 0, np.pi / 2, 3],[0.5, 0, 1, 0, 0, 4]])

    dfn_initial = np.asarray(
        [[-0.8, 0.4, 0, 0, np.pi / 2.2, 1.6, 1, 30, 5],
         [0.4, 0, 0.4, 0, np.pi / 3, 1.2, 0.8, 0, 5],
         [0, 0, -0.8, 0, np.pi / 3, 1.2, 1, 0, 5],
         [-0.8, -0.4, -0.88, 0, 0, 1.2, 0.7, 0, 5],
         [0.72, -0.6, 0.2, np.pi / 6, np.pi * 0.1, 1, 0.6, 0, 5]])

    sig_f = np.asarray([0, 0, 0, 1, 1])
    sig_v = np.asarray([0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0])
    sig_obs = 0.1

    low_bound = [0, -domainSize[0]/2, -domainSize[1]/2, -domainSize[2]/2, 0, 0, 0]
    high_bound = [10, domainSize[0]/2, domainSize[1]/2, domainSize[2]/2, np.pi, np.pi, 10]

    flow_condition = [['left', 7, 'right', 1], ['top', 6, 'bottom', 1], ['front', 5, 'back', 1]]

    mv = 'D'

    if not os.path.isdir(case_dir):
        os.mkdir(case_dir)

    # Run synthetic model
    synthetic_job = case_dir + '/synthetic'
    syn_engine = DFNINVERSE(synthetic_job, stations, domainSize, ncpu, flow_condition=flow_condition, relative_meshsize=100)
    syn_data = syn_engine.run_forward(dfn_synthetic)

    syn_data += normal(0, sig_obs, syn_data.shape)
    #
    with open(case_dir + '/forward_result.pkl', 'wb') as f:
        pickle.dump(syn_data, f)
    #
    syn_engine.gen_3d_obs_points_plot(stations, radius=np.average(domainSize) / 100)
    #
    # # Run inversion
    inverse_job = case_dir + '/inverse'

    dfninv = DFNINVERSE(inverse_job, stations, domainSize, ncpu, flow_condition=flow_condition)

    sp = MCMCSampler(dfninv, syn_data,
                     var_sigma=[sig_f, sig_v], moves=mv, prior_range=[low_bound, high_bound])

    chain = sp.sample(dfn_initial, max_iter, dodr)









