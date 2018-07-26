from dfninverse import DFNINVERSE
from sampler import MCMCSampler
from numpy.random import normal
import numpy as np
import pickle
from helper import stdout_redirect

if __name__ == '__main__':

    ncpu = 4

    case_dir = '/cluster/scratch/lishi/case_test'

    domainSize = [5.0, 5.0, 5.0]

    stations = [(2, 2, 1), (2, 2, -1), (2, -2, 1), (2, -2, -1), (-0.75, -0.4, 0.9), (-0.75, -0.4, 0)]

    dfn_synthetic = np.asarray(
        [[-2, 0, 0, 0, np.pi / 2, 4],[1.5, 0, 1, 0, np.pi / 2, 4],
         [2, 0, -1, 0, np.pi / 2, 4],[0.5, 0, 1, 0, 0, 4]])

    dfn_initial = np.asarray(
        [[-2, 0, 0, 0, np.pi / 2, 4],[1.5, 0, 1, 0, np.pi / 2, 4],
         [2, 0, -1, 0, np.pi / 2, 4],[1.5, 0, 1, 0, 0, 4]])

    sig_f = np.asarray([0, 0, 0, 1])
    sig_v = np.asarray([0.5, 0, 0, 0, 0, 0])
    sig_obs = 0.1

    low_bound = [0, -domainSize[0]/2, -domainSize[1]/2, -domainSize[2]/2, 0, 0, 0]
    high_bound = [10, domainSize[0]/2, domainSize[1]/2, domainSize[2]/2, np.pi, np.pi, 10]

    max_iter = 1000

    dodr = 5

    flow_condition = [['front', 5, 'back', 1]]

    mv = 'D'

    # Run synthetic model
    synthetic_job = case_dir + '/synthetic'
    syn_engine = DFNINVERSE(synthetic_job, stations, domainSize, ncpu, flow_condition=flow_condition)
    syn_data = syn_engine.run_forward(dfn_synthetic)

    print(syn_data)

    syn_data += normal(0, sig_obs, syn_data.shape)

    with open(case_dir + '/forward_result.pkl', 'wb') as f:
        pickle.dump(syn_data, f)

    syn_engine.gen_3d_obs_points_plot(stations, radius=np.average(domainSize) / 50)

    # Run inversion
    inverse_job = case_dir + '/inverse'

    dfninv = DFNINVERSE(inverse_job, stations, domainSize, ncpu, flow_condition=flow_condition)

    sp = MCMCSampler(dfninv, syn_data,
                     var_sigma=[sig_f, sig_v], moves=mv, prior_range=[low_bound, high_bound])

    chain = sp.sample(dfn_initial, max_iter, dodr)









