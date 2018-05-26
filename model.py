from dfninverse import DFNINVERSE
import numpy as np
import pandas as pd
from numpy.random import randint, uniform, normal, random
from numpy.linalg import det, inv, norm


class Variables:
#
    def __init__(self, **kwargs):

        # define the range of the value interval
        self.n_range = kwargs.get('n_range', [0, 10])
        self.cx_range = kwargs.get('x_range', [-0.5, 0.5])
        self.cy_range = kwargs.get('y_range', [-0.5, 0.5])
        self.cz_range = kwargs.get('z_range', [-0.5, 0.5])
        self.angle_range = kwargs.get('angle_range', [0, np.pi])
        self.radius_range = kwargs.get('radius_range', [0.5, 1])

        # define the number of fractures
        self.n = kwargs.get('n',
                            randint(self.n_range[0], self.n_range[1]))
        self.cx = kwargs.get('cx',
                             uniform(self.cx_range[0], self.cx_range[1], [self.n, 1]))
        self.cy = kwargs.get('cy',
                             uniform(self.cy_range[0], self.cy_range[1], [self.n, 1]))
        self.cz = kwargs.get('cz',
                             uniform(self.cz_range[0], self.cz_range[1], [self.n, 1]))
        self.phi = kwargs.get('phi',
                              uniform(self.angle_range[0], self.angle_range[1], [self.n, 1]))
        self.psi = kwargs.get('psi',
                              uniform(self.angle_range[0], self.angle_range[1], [self.n, 1]))
        self.r = kwargs.get('r',
                            uniform(self.radius_range[0], self.radius_range[1], [self.n, 1]))

        self.value = np.hstack((self.cx, self.cy, self.cz, self.phi, self.psi, self.r))


class ShapeVariables:

    def __init__(self, seed, sigma_f, sigma_v):

        if sigma_f.size == 1:
            self.sigma_f = sigma_f * np.eye(seed.shape[0])
        else:
            self.sigma_f = sigma_f

        if sigma_v.size == 1:
            self.sigma_v = sigma_v * np.eye(seed.shape[0])
        else:
            self.sigma_v = sigma_v

        self.seed = seed

        self.current_state = self.sigma_f @ self.seed @ self.sigma_v

    def deformation(self, current_state, U):

        self.deformed = self.sigma_f @ U @ self.sigma_v + current_state

        return self.deformed


class Matrix_normal_distribution:

    def __init__(self, M, U, V):

        self.n = M.shape[0]
        self.p = M.shape[1]

        self.M, self.U, self.V = (M, U, V)

    def get_density(self, X):

        denominator = np.sqrt((2 * np.pi) ** (self.n * self.p) * det(self.V) ** self.n * det(self.U) ** self.p)
        tr = np.trace(inv(self.V) @ (X - self.M).T @ inv(self.U) @ (X - self.M))
        numerator = np.exp(-0.5 * tr)

        print(tr)

        prob_density = numerator / denominator
        return prob_density

    def get_proposal(self):

        A = normal(0, 1, [self.n, self.p])

        X_new = self.M + self.U @ A @ self.V

        return X_new


def likelihood_ratio(syn1, syn2, obs, **kwargs):

    sig = kwargs.get('sig', 0.1) ** 2

    if syn1 is None:
        syn1 = np.zeros_like(obs)

    if syn2 is None:
        syn2 = np.zeros_like(obs)

    # lh_1 = norm((syn1-obs)/sig)
    # lh_2 = norm((syn2-obs)/sig)
    #
    # print(lh_1, lh_2)
    # ll_ratio = np.exp(0.5*(lh_1 - lh_2))
    n_station = obs.shape[1]
    n_timestep = obs.shape[0]

    st = kwargs.get('sig_time',  0.01*np.ones(n_timestep))
    ss = kwargs.get('sig_station', np.ones(n_station))

    sigma_time = np.diag(st)
    sigma_station = np.diag(ss)

    tr1 = np.trace(inv(sigma_station) @ (syn1 - obs).T @ inv(sigma_time) @ (syn1 - obs)) / (n_station * n_timestep)
    tr2 = np.trace(inv(sigma_station) @ (syn2 - obs).T @ inv(sigma_time) @ (syn2 - obs)) / (n_station * n_timestep)
    # print(syn1-obs)
    print(tr1, tr2, tr1-tr2)
    # # likelihood = Matrix_normal_distribution(obs, sigma_time, sigma_station).get_density(syn)
    ll_ratio = np.exp(-0.5*(tr2-tr1)) / np.sqrt(2*np.pi)
    print(ll_ratio)
    return ll_ratio


if __name__ == '__main__':

    obs_points = [(0.4, 0.4, 0.2),
                  (0.4, 0.4, -0.2),
                  (0.4, -0.4, 0.2),
                  (0.4, -0.4, -0.2),
                  (-0.15, -0.08, 0.2),
                  (-0.15, -0.08, 0)]

    dfninv = DFNINVERSE('/Volumes/SD_Card/Thesis_project/dfn_test',
                        '/Volumes/SD_Card/Thesis_project/synthetic_model/output/obs_readings.csv',
                        obs_points
                        )

    obs_data = dfninv.obs_data
    # obs_fractures = pd.read_csv('/Volumes/SD_Card/Thesis_project/synthetic_model/inputs/observed_fractures.csv').values

    obs_fractures = np.asarray([[-0.4, 0, 0, 0, np.pi/2, 0.85],
                                [0.3, 0, 0.2, 0, np.pi/2, 0.8],
                                [0.4, 0, -0.2, 0, np.pi/2, 0.8]])
    unknown_fractures = np.asarray([
                                   [0, 0, 0, 0, 0, 0.6],
                                   ])
    n_unknown_frac = 1
    n_observed_frac = obs_fractures.shape[0]

    n = n_unknown_frac + n_observed_frac

    # Define covariance matrix \Sigma_f

    sig_obs = 0.001
    sig_unknown = 1

    sig_f = np.hstack((sig_obs * np.ones(n_observed_frac), sig_unknown * np.ones(n_unknown_frac)))
    sigma_f = np.diag(sig_f)

    # Define covariance matrix \Sigma_v
    sig_v = 1
    sigma_v = np.diag(sig_v * np.ones(6))

    # Define Initial shape variables

    # shape_variables = Variables(n=n)
    # s_current = shape_variables.value
    # s_current[0:n_observed_frac] = obs_fractures

    s_current = np.vstack((obs_fractures, unknown_fractures))
    # print(s_current)

    # Initialize MCMC
    dfninv.run_forward_simulation(s_current)
    dfninv.save_accepted_model(model_id=0)
    syn_data_old = dfninv.read_simulation_results()
    # print(obs_data)
    # print(syn_data_old)
    # print(syn_data_old - obs_data)
    # likelihood_ratio(syn_data_old, syn_data_old, obs_data)
    # likelihood_old = likelihood_function(syn_data_old, obs_data)
    prior_old = 1
    #
    # Start loop
    max_iteration = 3000
    i_iter = 0
    accept_case = 0
    while i_iter < max_iteration:
        print('{0}Iteration:{1:d}{2}'.format('*'*20, i_iter, '*'*20))
        accept_flag = False

        state = 'new'
        s_new = Matrix_normal_distribution(s_current, sigma_f, sigma_v).get_proposal()
        dfninv.run_forward_simulation(s_new)
        syn_data_new = dfninv.read_simulation_results()
        # likelihood_new = likelihood_function(syn_data_new, obs_data)
        prior_new = 1

        ratio_likelihood = likelihood_ratio(syn_data_old, syn_data_new, obs_data)
        ratio_prior = prior_new / prior_old
        ratio_proposal = 1 #Matrix_normal_distribution(s_current, sigma_f, sigma_v).get_density(s_new)

        # calculate acceptance ratio
        Jacob = 1
        alpha_1 = min(1, ratio_likelihood * ratio_prior * ratio_proposal * Jacob)
        print('Liklihood ratio = {:.2f}'.format(ratio_likelihood))
        print('Acceptance Ratio = {:3f}'.format(alpha_1))

        if alpha_1 > random():
            accept_flag = True
            accept_case += 1

            s_current = s_new
            prior_old = prior_new
            # likelihood_old = likelihood_new
            print('### Accept! ###')
            dfninv.save_accepted_model(model_id=accept_case)

        i_iter += 1








