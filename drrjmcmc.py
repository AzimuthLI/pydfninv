"""
.. module:: DRRJ_MCMC.py
   :synopsis: Delayed Rejection REVERSIBEL JUMPY MCMC
.. moduleauthor:: Shiyi Li

"""

import numpy as np
import numpy.random as rd
from distribution import Normal_distribution
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white')
sns.set_context('talk')

def run_simulation(vars):

    x=np.linspace(0, 10, 100)
    sim_readings = vars['A'] * x + vars['B']

    return sim_readings

def gaussian_likely(y, y_obs, sigma=5):
    '''

    :param y:
    :param y_obs:
    :param sigma:
    :return:
    '''
    const = 1/(np.sqrt(2*np.pi)*sigma)
    likelyhood = const * np.exp(-0.5 * np.sqrt(np.sum((y - y_obs) ** 2)) / (2 * sigma ** 2))
    return likelyhood

class DRRJ_MCMC:

    def __init__(self, obs, vars, **kwargs):

        self.dodr = kwargs.get('dodr', True)
        self.n_step = kwargs.get('n_step', 5000)

        self.obs = obs
        self.vars = vars # list of variable objects

    def sample(self, **kwargs):

        start = kwargs.get('start', [])

        if start == []:
            old_vars = self.vars.copy()
            for var in self.vars:
                old_vars[var] = self.vars[var].make_proposal(rd.normal(0,1))
        else:
            old_vars = start
        print(old_vars)
        i_iter = 0
        acce = 0
        trace = []

        while i_iter < self.n_step:

            accept = 0

            old_Y_simulate = run_simulation(old_vars)
            old_likely = gaussian_likely(old_Y_simulate, self.obs)
            old_prior = gaussian_likely(old_vars['A'], 3, sigma=1) * gaussian_likely(old_vars['B'], 6, sigma=1)
            # old_likely = log_likelihood(old_Y_simulate, self.obs)

            new_vars = self.vars.copy()

            for var in self.vars:
                new_vars[var] = self.vars[var].make_proposal(old_vars[var])

            new_Y_simulate = run_simulation(new_vars)
            new_likely = gaussian_likely(new_Y_simulate, self.obs)
            new_prior = gaussian_likely(new_vars['A'], 3, sigma=1) * gaussian_likely(new_vars['B'], 6, sigma=1)
            # new_likely = log_likelihood(new_Y_simulate, self.obs)
            # new_rms = rms(new_Y_simulate, self.obs)

            alpha = min(1, new_likely*new_prior/(old_likely*old_prior))

            # alpha = min([1, np.exp((old_rms**2 - new_rms**2)/(2*15**2))])

            beta = rd.random()
            if alpha > beta:
                accept = 1
                acce += 1
                old_vars = new_vars
                trace.append(old_vars)

            i_iter +=1
            if i_iter%5000== 0:
                print('new_rms= %.10f, alpha=%.2f, acc = %d, iter =%d' % (new_likely, alpha, accept, i_iter))

        return trace

if __name__ == "__main__":

    fixed_vars = {'A': 3, 'B': 6}

    Y_obs = run_simulation(fixed_vars) + rd.normal(0, 2, 100)

    # plt.plot(Y_obs, '.')
    # plt.show()

    var_A = Normal_distribution(mu=3, std=10)
    var_B = Normal_distribution(mu=5.5, std=10)


    vars = {'A': var_A, 'B': var_B}

    mcmc = DRRJ_MCMC(Y_obs, vars)

    trace = mcmc.sample()

    A = []
    B = []
    for vars in trace:
        A.append(vars['A'])
        B.append(vars['B'])

    plt.figure(figsize=[16, 12])
    plt.subplot(221)
    plt.plot(A, 'b', fixed_vars['A']*np.ones_like(A),'r')

    plt.subplot(222)
    plt.plot(B, 'b', fixed_vars['B'] * np.ones_like(A), 'r')

    plt.subplot(223)
    sns.distplot(A, norm_hist=True)
    # sns.distplot(np.random.normal(3, 2, 50))
    plt.plot([3, 3], [0,1])
    plt.subplot(224)
    sns.distplot(B, norm_hist=True)
    plt.plot([6, 6], [0, 1], '-')
    # sns.distplot(np.random.normal(6, 2, 50))
    plt.show()


    plt.show()

