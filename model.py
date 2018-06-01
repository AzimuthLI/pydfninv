import numpy as np
from dfninverse import DFNINVERSE
from numpy.random import randint, uniform, normal, random
from numpy.linalg import det, inv, norm
import matplotlib.pyplot as plt
from numpy import diff
from helper import latexify


def prior_probability(s, **kwargs):
    # define the range of the value interval
    n_range = kwargs.get('n_range', np.asarray([0, 10]))
    center_range = kwargs.get('x_range', np.asarray([[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]]))
    angle_range = kwargs.get('angle_range', np.asarray([[0, np.pi], [0, np.pi]]))
    radius_range = kwargs.get('radius_range', np.asarray([0.5, 1]))

    lower_bound = np.asarray([center_range[0, 0],
                            center_range[1, 0],
                            center_range[2, 0],
                            angle_range[0, 0],
                            angle_range[1, 0],
                            radius_range[0]])

    upper_bound = np.asarray([center_range[0, 1],
                            center_range[1, 1],
                            center_range[2, 1],
                            angle_range[0, 1],
                            angle_range[1, 1],
                            radius_range[1]])

    n_total = s.shape[0]

    if (n_total < n_range[1]) & \
            (n_total > n_range[0]) & \
            np.all((s - lower_bound) >= 0) & \
            np.all((s - upper_bound) <= 0):

        prior = (1 / diff(n_range)) * \
                (diff(angle_range[0, :]) * diff(angle_range[1, :]) *
                 diff(center_range[0, :]) * diff(center_range[1, :]) * diff(center_range[2, :]) *
                 diff(radius_range)
                 ) ** (-n_total)
    else:
        prior = 0

    return prior


def likelihood_ratio(syn1, syn2, obs, **kwargs):

    if syn1 is None:
        syn1 = np.zeros_like(obs)

    if syn2 is None:
        syn2 = np.zeros_like(obs)

    n_station = obs.shape[1]
    n_timestep = obs.shape[0]

    st = kwargs.get('sig_time',  0.01*np.ones(n_timestep))
    ss = kwargs.get('sig_station', np.ones(n_station))

    sigma_time = np.diag(st)
    sigma_station = np.diag(ss)

    tr1 = np.trace(inv(sigma_station) @ (syn1 - obs).T @ inv(sigma_time) @ (syn1 - obs)) / (n_station * n_timestep)
    tr2 = np.trace(inv(sigma_station) @ (syn2 - obs).T @ inv(sigma_time) @ (syn2 - obs)) / (n_station * n_timestep)

    ll_ratio = np.exp(-0.5*(tr2-tr1)) / np.sqrt(2*np.pi)
    return tr1, tr2, ll_ratio


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

        A = normal(0, 1, self.M.shape)
        X_proposed = self.M + self.U @ A @ self.V

        return X_proposed

if __name__ == '__main__':

    # Define project information
    station_coordinates = [(0.4, 0.4, 0.2),   (0.4, 0.4, -0.2),    (0.4, -0.4, 0.2),
                           (0.4, -0.4, -0.2), (-0.15, -0.08, 0.2), (-0.15, -0.08, 0)]

    project_path = '/Volumes/SD_Card/Thesis_project/dfn_test'
    field_observation_file = '/Volumes/SD_Card/Thesis_project/synthetic_model/output/obs_readings.csv'
    ncpu = 1

    dfninv = DFNINVERSE(project_path, field_observation_file, station_coordinates, ncpu)
    field_observation = dfninv.obs_data

    # Define the initial fractures
    observed_fractures = np.asarray([[-0.4, 0, 0, 0, np.pi / 2, 0.85],
                                     [0.3, 0, 0.2, 0, np.pi / 2, 0.8],
                                     [0.4, 0, -0.2, 0, np.pi / 2, 0.8]])
    inferred_fractures = np.asarray([[0, 0, 0, np.pi / 2, 0, 0.6]])

    n_inferred_frac = inferred_fractures.shape[0]
    n_observed_frac = observed_fractures.shape[0]

    n_total = n_inferred_frac + n_observed_frac

    s_current = np.vstack((observed_fractures, inferred_fractures))

    # Define covariance matrix \Sigma_f, \Sigma_v
    sig_obs = 0.001
    sig_unknown = 1
    sig_v = 0.05

    sig_f = np.hstack((sig_obs * np.ones(n_observed_frac), sig_unknown * np.ones(n_inferred_frac)))
    sigma_f = np.diag(sig_f)
    sigma_v = np.diag(sig_v * np.ones(6))

    # Initialize MCMC
    prior_current = prior_probability(s_current)

    dfninv.run_forward_simulation(s_current)
    syn_data_current = dfninv.read_simulation_results()
    rms_0, rms_1, ratio_0 = likelihood_ratio(syn_data_current, syn_data_current, field_observation)
    status = {'model_id': 0, 'RMS': rms_0, 'fractures': s_current}
    dfninv.save_accepted_model(status, model_id=0, save_flag=True)

    # Start loop
    max_iteration = 10
    i_iter = 0
    accept_case = 0

    while i_iter < max_iteration:
        print('{0}Iteration:{1:d}{2}'.format('*' * 20, i_iter, '*' * 20))
        accept_flag = False

        s_proposed = Matrix_normal_distribution(s_current, sigma_f, sigma_v).get_proposal(A)
        dfninv.run_forward_simulation(s_proposed)
        syn_data_proposed = dfninv.read_simulation_results()
        prior_proposed = prior_probability(s_proposed)

        rms_current, rms_proposed, ratio_ll = likelihood_ratio(syn_data_current, syn_data_proposed, field_observation)
        ratio_prior = prior_proposed / prior_current
        ratio_proposal = 1

        # calculate acceptance ratio
        alpha_1 = min(1, ratio_ll * ratio_prior * ratio_proposal)
        print('Likelihood ratio = {:.2f}'.format(ratio_ll))
        print(alpha_1)
        # print('Acceptance Ratio = {:3f}'.format(alpha_1))

        if alpha_1 > random():
            accept_flag = True
            accept_case += 1

            s_current = s_proposed
            prior_current = prior_proposed
            rms_current = rms_proposed
            print('### Accept! RMS = {} ###'.format(rms_current))

        status = {'model_id': accept_case, 'RMS': rms_current, 'fractures': s_current}
        dfninv.save_accepted_model(status, model_id=accept_case, save_flag=accept_flag)
        i_iter += 1


    # Analyze result
    with open('/Volumes/SD_Card/Thesis_project/dfn_test/mcmc_log.txt', 'r') as logfile:
        mcmc_log = logfile.readlines()
        separator = '*' * 60 + '\n'
        idx_sep = [i for i, j in enumerate(mcmc_log) if j == separator]

        n_fractures = idx_sep[1] - idx_sep[0] - 3
        n_model = len(idx_sep)

        rms = []
        for i in idx_sep:
            rms.append(float(mcmc_log[i+2].replace('\n', '')))

    latexify()
    fig = plt.figure()
    plt.plot(rms)
    plt.xlabel('Iteration')
    plt.ylabel('RMS')
    plt.title('Root-Mean-Square Error Evolution as MCMC Iteration')
    plt.savefig(project_path + '/rms_iteration.png')
    plt.show()















