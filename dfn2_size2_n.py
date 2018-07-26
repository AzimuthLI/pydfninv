from dfninverse import DFNINVERSE
from sampler import MCMCSampler
from numpy.random import normal
import numpy as np
import pickle, os
from helper import stdout_redirect

if __name__ == '__main__':

    ncpu = 7

    case_dir = '/cluster/scratch/lishi/dfn2_size2_n'
    max_iter = 2000
    dodr = 2
    mv = ['D', 'B', 'K']

    if not os.path.isdir(case_dir):
        os.mkdir(case_dir)

    domainSize = [2.0, 2.0, 2.0]

    stations = [(0.4, 0, -0.77), (0.3, 0.5, -0.77), (-0.5, -0.3, -0.84),
                (-0.5, -0.3, -0.4),(0.1, 0.5, -0.4),(0.3, -0.7, -0.4),
                (0.6, 0.3, -0.4), (0.3, 0, 0.18), (-0.1, 0, 0.2),
                (-0.5, 0.3, 0.5), (0.5, -0.3, 0.3)]

    # Setup dfn : | x | y | z | phi | psi | radius | aspect_ratio | beta | n_vertices |
    dfn_synthetic = np.asarray(
        [[0, 0, 0.2, 0.1, np.pi/2.1, 1, 0.6, 60, 10], #obs-horizontal
         [0, 0, 0.6, 0, np.pi/2.3, 1, 0.7, 20, 10], #obs-horizontal
         [0, 0, -0.8, 0, np.pi/1.9, 1, 0.8, 0, 10], #obs-horizontal
         [0, 0, -0.4, 0, np.pi/2, 1.2, 1, 0, 10], #obs-horizontal
         [0, 0, -0.2, 0.1, np.pi/2.1, 1, 0.6, 0, 10],   #infer-vertical, missing
         [0, 0.1, 0.3, np.pi/6, np.pi/9, 1.2, 0.4, 0, 10],  #infer-vertical
         [-0.2, 0, -0.4, np.pi/6, np.pi/9, 0.8, 1, 0, 10]]) #infer-vertical

    dfn_initial = np.asarray(
        [[0,   0,     0.2,    0.1, np.pi / 2.1, 1, 0.6, 60, 10], # obs-horizontal
         [0,   0,     0.4,  0,   np.pi / 2.3, 1, 0.7, 20, 10],  # obs-horizontal
         [0,   0,    -0.8,  0,   np.pi / 1.9, 1, 0.8, 0, 10],   # obs-horizontal
         [0,   0,    -0.4,  0,   np.pi / 2, 1.2, 1, 0, 10],     # obs-horizontal
         [0, 0.3,  1,   np.pi / 6, np.pi / 9, 1.2, 0.4, 0, 10],      # infer-vertical
         [0.4, -0.5, -0.4,  np.pi / 6, np.pi / 9, 0.8, 1, 0, 10]])  # infer-vertical

    reference_dfn = np.asarray([0, 0, 0, 0.1, np.pi / 2.1, 1, 0.6, 60, 10])

    sig_f = np.asarray([0, 0, 0, 0, 1, 1])
    sig_v = np.asarray([0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0])
    sig_obs = 0.1

    low_bound = [0, -domainSize[0]/2, -domainSize[1]/2, -domainSize[2]/2, 0, 0, 0]
    high_bound = [10, domainSize[0]/2, domainSize[1]/2, domainSize[2]/2, np.pi, np.pi, 10]

    flow_condition = [['front', 5, 'back', 1], ['left', 5, 'right', 1], ['top', 5, 'bottom', 1]]

    # Run synthetic model
    synthetic_job = case_dir + '/synthetic'
    syn_engine = DFNINVERSE(synthetic_job, stations, domainSize, ncpu, flow_condition=flow_condition, relative_meshsize=200)
    syn_data = syn_engine.run_forward(dfn_synthetic)

    syn_data += normal(0, sig_obs, syn_data.shape)

    with open(case_dir + '/forward_result.pkl', 'wb') as f:
        pickle.dump(syn_data, f)

    syn_engine.gen_3d_obs_points_plot(stations, radius=np.average(domainSize) / 50)

    # Run inversion
    inverse_job = case_dir + '/inverse'

    dfninv = DFNINVERSE(inverse_job, stations, domainSize, ncpu, flow_condition=flow_condition)

    sp = MCMCSampler(dfninv, syn_data,
                     var_sigma=[sig_f, sig_v], moves=mv,
                     prior_range=[low_bound, high_bound],
                     reference=reference_dfn)

    chain = sp.sample(dfn_initial, max_iter, dodr)









