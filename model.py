import numpy as np
from dfninverse import DFNINVERSE
from numpy.random import randint, normal, random
from numpy.linalg import det, inv
from numpy import diff
from scipy import stats
import pickle

class matrix_normal:

    def __init__(self, M, U, V):

        self.n = M.shape[0]
        self.p = M.shape[1]

        self.M, self.U, self.V = (M, U, V)

    def pdf(self, X):

        denominator = np.sqrt((2 * np.pi) ** (self.n * self.p) * det(self.V) ** self.n * det(self.U) ** self.p)
        try:
            tr = np.trace(inv(self.V) @ (X - self.M).T @ inv(self.U) @ (X - self.M))
        except:
            r_idx = np.all(self.V, axis=1)
            c_idx = np.all(self.U, axis=0)
            v = self.V[r_idx].T[r_idx]
            u = self.U[c_idx].T[c_idx]
            x = X[np.where(r_idx), np.where(c_idx)]
            m = self.M[np.where(r_idx), np.where(c_idx)]
            print(v)
            print(u)
            if v.shape == 1:
                v_inv = 1/v
            else:
                v_inv = inv(v)
            if u.ndim == 1:
                u_inv = 1/u
            else:
                u_inv = inv(u)

            tr = np.trace(v_inv @ (x-m).T @ u_inv @ (x-m))

        numerator = np.exp(-0.5 * tr)

        prob_density = numerator / denominator
        return prob_density


class mcmc_sampler:

    def __init__(self, inverse_engine, observation, **kwargs):

        self.prior_range = kwargs.get('prior_range', None)
        self.obs_sigma = kwargs.get('obs_sigma', None)
        self.move_list = kwargs.get('moves', ['D', 'B', 'K'])
        var_sigma = kwargs.get('var_sigma', None)

        self.sigma_fracture = np.diag(var_sigma[0])
        self.sigma_shape = np.diag(var_sigma[1])

        self.observation = observation
        self.engine = inverse_engine

        self.move = None
        self.chain_length = None
        self.dr_scale = None
        self.state = None
        self.accept_case = 0
        self.save_flag = True

    def get_state(self, dfn):
        return {'dfn_id': self.accept_case,
                'dfn': dfn,
                'prior': self.prior_func(dfn),
                'rms': self.misfit_func(dfn)}

    def prior_func(self, dfn):

        if self.prior_range is None:
            lower_bound = np.asarray([0, -0.5, -0.5, -0.5, 0, 0, 0.1])
            upper_bound = np.asarray([10, 0.5, 0.5, 0.5, np.pi, np.pi, 0.9])
        else:
            lower_bound = self.prior_range[0]
            upper_bound = self.prior_range[1]

        prior = stats.uniform.pdf(dfn.shape[0],
                                  loc=lower_bound[0],
                                  scale=upper_bound[0])

        if np.all((dfn - lower_bound[1:]) >= 0) &\
                np.all((dfn - upper_bound[1:]) <= 0):
            p = diff([lower_bound[1:], upper_bound[1:]], axis=0)
            prior *= np.prod(p) ** (- dfn.shape[0])
        else:
            prior *= 0

        return prior

    def misfit_func(self, dfn):

        n_station = self.observation.shape[1]
        n_timestep = self.observation.shape[0]

        if self.obs_sigma is None:
            st = 0.01 * np.ones(n_timestep)
            ss = np.ones(n_station)

            sigma_time = np.diag(st)
            sigma_station = np.diag(ss)
        else:
            sigma_time = self.obs_sigma['time']
            sigma_station = self.obs_sigma['station']

        syn = self.engine.run_forward(dfn)
        obs = self.observation
        if syn is None:
            syn = np.zeros_like(obs)
        rms = np.trace(inv(sigma_station) @ (syn - obs).T @ inv(sigma_time) @ (syn - obs)) / (n_station * n_timestep)

        return rms

    def step(self):

        self.move = self.move_list[randint(0, len(self.move_list))]
        print('Move Type: {}'.format(self.move))
        state_1 = self.propose(1)
        accept_flag = self.acceptance_condition(state_1)
        if accept_flag:
            self.state = state_1
        elif (self.dr_scale != 0) & (self.move == 'D'):
            state_2 = self.propose(2)
            state2_1 = self.propose(-1, st = state_2)
            accept_flag = self.acceptance_condition(state_1, state_2, state2_1)
            if accept_flag:
                self.state = state_2
        return accept_flag

    def sample(self, initial_dfn, chain_length=100, dr_scale=0):

        self.chain_length = chain_length
        self.dr_scale = dr_scale

        self.state = self.get_state(initial_dfn)
        chain = [self.state]
        self.save_sample()

        i = 0

        while i < self.chain_length:
            print('{0}Iteration:{1:d}{0}'.format('*' * 30, i))
            self.save_flag = self.step()
            if self.save_flag:
                self.accept_case += 1

            self.save_sample()
            chain.append(self.state)

            print('Accept: {}'.format(self.save_flag))
            print('Current RMS={:.2f}'.format(self.state['rms']))
            i += 1

        return chain

    def acceptance_condition(self, *args):

        if len(args) == 1:
            state1 = args[0]
            alp = self.alpha_1(state1)
        elif len(args) == 3:
            s1, s2, s2_1 = args
            alp = self.alpha_2(s1, s2, s2_1)
        else:
            raise ValueError('Unresolved arguments')
        print('Alpha = {:.4f}'.format(alp))
        if alp > random():
            accept_flag = True
        else:
            accept_flag = False

        return accept_flag

    def alpha_2(self, *args):
        s1, s2, s2_1 = args
        a1 = self.alpha_1(s1)
        a2_1 = self.alpha_1(s2, s2_1)
        a2 = self.alpha_1(s2)

        sig_f = self.sigma_fracture / self.dr_scale
        sig_v = self.sigma_shape / self.dr_scale

        q2_1 = matrix_normal(s2['dfn'], sig_f, sig_v).pdf(s2_1['dfn'])
        q1 = matrix_normal(self.state['dfn'], self.sigma_fracture, self.sigma_shape).pdf(s1['dfn'])

        alpha_2 = min(1, (q2_1/q1) * a2 * (1-a2_1)/(1-a1))

        return alpha_2

    def alpha_1(self, *args):

        if len(args) == 1:
            s0 = self.state
            s1 = args[0]
        elif len(args) == 2:
            s0, s1 = args
        else:
            raise ValueError('Unresolved arguments')

        rms_0 = s0['rms']
        prior_0 = s0['prior']
        dfn_0 = s0['dfn']

        rms_1 = s1['rms']
        prior_1 = s1['prior']
        dfn_1 = s1['dfn']

        ll_ratio = np.exp(-0.5 * (rms_1 - rms_0)) / np.sqrt(2 * np.pi)
        prior_ratio = prior_1 / prior_0
        proposal_ratio = self.proposal_ratio(dfn_0, dfn_1)
        alpha_1 = min(1, ll_ratio * prior_ratio * proposal_ratio)

        return alpha_1

    def proposal_ratio(self, dfn0, dfn1):

        if self.move == 'D':
            q = 1
        else:
            if dfn1.shape[0] < dfn0.shape[0]:
                t = dfn1
                dfn1 = dfn0
                dfn0 = t
            n_vars = dfn1.shape[1]
            n_fractures = dfn1.shape[0]

            coeff = 1/np.sqrt((np.pi * 2) ** n_vars * det(self.sigma_shape))
            delta = dfn1[-1, :] - np.average(dfn0, axis=0)

            q_birth = np.exp(-0.5 * delta.T @ inv(self.sigma_shape) @ delta) / coeff
            q_kill = 1 / n_fractures
            if self.move == 'B':
                q = q_kill / q_birth
            elif self.move == 'K':
                q = q_birth / q_kill

        return q

    def save_sample(self):
        self.engine.write_inverselog(self.state, model_id=self.accept_case, save_flag=self.save_flag)
        with open(self.engine.project + '/mcmc_chain.pkl', 'ab') as f:
            pickle.dump(self.state, f)

    def propose(self, step_stage, **kwargs):
        dfn_old = self.state['dfn']
        n_frac = dfn_old.shape[0]
        n_var = dfn_old.shape[1]

        if step_stage == 1:
            if self.move == 'D':
                A = normal(0, 1, dfn_old.shape)
                dfn_new = dfn_old + self.sigma_fracture @ A @ self.sigma_shape

            elif self.move == 'B':
                A = normal(0, 1, [1, n_var])
                dfn_append = np.average(dfn_old, axis=0) + (A @ self.sigma_shape).T
                dfn_new = np.hstack((dfn_old, dfn_append))

            elif self.move == 'K':
                idx = randint(0, n_frac)
                dfn_new = np.delete(dfn_old, idx, axis=0)

        if step_stage == 2:
            sig_f = self.sigma_fracture / self.dr_scale
            sig_v = self.sigma_shape / self.dr_scale

            A = normal(0, 1, dfn_old.shape)
            dfn_new = dfn_old + sig_f @ A @ sig_v

        if step_stage == -1:
            state = kwargs.get('st')
            dfn_old = state['dfn']
            A = - normal(0, 1, dfn_old.shape)
            dfn_new = dfn_old + self.sigma_fracture @ A @ self.sigma_shape

        s_new = self.get_state(dfn_new)
        print('RMS of proposal: {:.3f}'.format(s_new['rms']))
        return s_new


if __name__ == '__main__':
    # Set inverse project information

    # Model 1 (1X1X1， cx)
    # dsize = [1.0, 1.0, 1.0]
    # station_coordinates = [(0.4, 0.4, 0.2), (0.4, 0.4, -0.2), (0.4, -0.4, 0.2),
    #                        (0.4, -0.4, -0.2), (-0.15, -0.08, 0.2), (-0.15, -0.08, 0)]
    # observed_fractures = np.asarray([[-0.4, 0, 0, 0, np.pi / 2, 0.8],
    #                                  [0.3, 0, 0.2, 0, np.pi / 2, 0.8],
    #                                  [0.4, 0, -0.2, 0, np.pi / 2, 0.8]])
    # inferred_fractures = np.asarray([[-0.2, 0, 0.2, 0, 0, 0.8]])

    # Model 2 (1X1X1, cxyz)
    dsize = [1.0, 1.0, 1.0]
    station_coordinates = [(-0.05, 0.2, 0.05), (-0.11, 0.4, 0.21), (-0.3, -0.2, 0.1), (0.2, -0.1, 0.4),
                           (-0.4, 0.2, -0.2), (0.2, -0.2, 0.2), (0.3, 0.2, -0.3)]
    observed_fractures = np.asarray([[-0.4, 0, 0,  0.7854, 0.6283, 0.8],
                                     [0.3, 0, 0.2, 0.7854, 0.6283, 0.8],
                                     [0.4, 0, -0.2, 0.7854, 0.6283, 0.8]])
    inferred_fractures = np.asarray([[0, 0, 0.2, 0, 2.356, 0.8]])

    # Model 3 (5X5X5， cx)
    # dsize = [5.0, 5.0, 5.0]
    # station_coordinates = [(2, 2, 1), (2, 2, -1), (2, -2, 1),
    #                        (2, -2, -1), (-0.75, -0.4, 1), (-0.75, -0.4, 0)]
    #
    # observed_fractures = np.asarray([[-2, 0, 0, 0, np.pi / 2, 4],
    #                                  [1.5, 0, 1, 0, np.pi / 2, 4],
    #                                  [2, 0, -1, 0, np.pi / 2, 4]])
    #
    # inferred_fractures = np.asarray([[1.5, 0, 1, 0, 0, 4]])

    # Model 4 (5X5X5， cxyz)
    # dsize = [2.0, 2.0, 2.0]
    # station_coordinates = [(-0.1, 0.4, 0.1), (-0.22, 0.8, 0.42), (-0.6, -0.4, 0.2), (0.4, -0.2, 0.8),
    #                        (-0.8, 0.4, -0.4), (0.4, -0.4, 0.4), (0.6, 0.4, -0.6)]
    # observed_fractures = np.asarray([[-0.8, 0, 0, 0.7854, 0.6283, 1.6],
    #                                  [0.6, 0, 0.4, 0.7854, 0.6283, 1.6],
    #                                  [0.8, 0, -0.4, 0.7854, 0.6283, 1.6]])
    # inferred_fractures = np.asarray([[0, 0, 0.4, 0, 2.356, 1.6]])

    # Initialize inverse engine
    project_path = '/cluster/home/lishi/model_1x1x1_cxyz_1000/inverse_2'
    field_observation_file = '/cluster/home/lishi/model_1x1x1_cxyz_1000/synthetic/output/obs_readings.csv'
    ncpu = 4

    dfninv = DFNINVERSE(project_path, field_observation_file, station_coordinates, dsize, ncpu)
    field_observation = dfninv.obs_data

    # Define the initial fractures
    n_inferred_frac = inferred_fractures.shape[0]
    n_observed_frac = observed_fractures.shape[0]

    # Define covariance matrix \Sigma_f, \Sigma_v
    sig_obs = 0
    sig_unknown = 1

    sig_f = np.hstack((sig_obs * np.ones(n_observed_frac), sig_unknown * np.ones(n_inferred_frac)))
    sig_v = np.asarray([0.01, 0.01, 0.01, 0, 0, 0])

    prior_range = [[0, -dsize[0]/2, -dsize[1]/2, -dsize[2]/2, 0, 0, 0],
                   [10, dsize[0]/2, dsize[1]/2, dsize[2]/2, np.pi, np.pi, 2]]

    sp = mcmc_sampler(dfninv, field_observation, var_sigma=[sig_f, sig_v], moves='D', prior_range=prior_range)

    s_initial = np.vstack((observed_fractures, inferred_fractures))

    chain = sp.sample(s_initial, chain_length=2000, dr_scale=1)

    # with open(project_path+'/mcmc_chain.pkl', 'wb') as f:
    #     pickle.dump(chain, f)
