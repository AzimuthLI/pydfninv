from dfninverse import DFNINVERSE
from sampler import MCMCSampler
from numpy.random import normal
import numpy as np
import pickle, os
from helper import stdout_redirect

if __name__ == '__main__':

    ncpu = 4

    case_dir = '/cluster/scratch/lishi/dfn3_size1_xyz_dr0'

    if not os.path.isdir(case_dir):
        os.mkdir(case_dir)

    domainSize = [1.0, 1.0, 1.0]

    stations = [(0.4, 0.4, 0.2), (0.4, 0.4, -0.2), (0.4, -0.4, 0.2),
                (0.4, -0.4, -0.2), (-0.15, -0.08, 0.2), (-0.15, -0.08, 0)]

    # Setup dfn : | x | y | z | phi | psi | radius | aspect_ratio | beta | n_vertices |
    dfn_synthetic = np.asarray(
        [[-0.4, 0, 0, 0, np.pi / 2, 0.8, 1, 0, 5],
         [0.3, 0, 0.2, 0, np.pi / 2, 0.8, 1, 0, 5],
         [0.4, 0, -0.2, 0, np.pi / 2, 0.8, 1, 0, 5],
         [0.1, 0, 0.2, 0, 0, 0.8, 1, 0, 5]])

    dfn_initial = np.asarray(
        [[-0.4, 0, 0, 0, np.pi / 2, 0.8, 1, 0, 5],
         [0.3, 0, 0.2, 0, np.pi / 2, 0.8, 1, 0, 5],
         [0.4, 0, -0.2, 0, np.pi / 2, 0.8, 1, 0, 5],
         [-0.2, 0.2, 0, 0, 0, 0.8, 1, 0, 5]])

    sig_f = np.asarray([0, 0, 0, 1])
    sig_v = np.asarray([0.05, 0.05, 0.05, 0, 0, 0, 0, 0, 0])
    sig_obs = 0.1

    low_bound = [0, -domainSize[0]/2, -domainSize[1]/2, -domainSize[2]/2, 0, 0, 0]
    high_bound = [10, domainSize[0]/2, domainSize[1]/2, domainSize[2]/2, np.pi, np.pi, 10]

    max_iter = 1000

    dodr = 0

    flow_condition = [['left', 3, 'right', 1], ['top', 6, 'bottom', 1], ['front', 5, 'back', 1]]

    mv = 'D'

    # Run synthetic model
    synthetic_job = case_dir + '/synthetic'
    syn_engine = DFNINVERSE(synthetic_job, stations, domainSize, ncpu, flow_condition=flow_condition, relative_meshsize=100)
    syn_data = syn_engine.run_forward(dfn_synthetic)

    syn_data += normal(0, sig_obs, syn_data.shape)

    with open(case_dir + '/forward_result.pkl', 'wb') as f:
        pickle.dump(syn_data, f)

    syn_engine.gen_3d_obs_points_plot(stations, radius=np.average(domainSize) / 100)

    # # Run inversion
    inverse_job = case_dir + '/inverse'

    dfninv = DFNINVERSE(inverse_job, stations, domainSize, ncpu, flow_condition=flow_condition)

    sp = MCMCSampler(dfninv, syn_data, obs_sigma=sig_obs,
                     var_sigma=[sig_f, sig_v], moves=mv, prior_range=[low_bound, high_bound])

    chain = sp.sample(dfn_initial, max_iter, dodr)









