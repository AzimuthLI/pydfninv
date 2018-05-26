
import pandas as pd
import numpy as np
from numpy.random import randint, random

def rms(state):

    observation = pd.read_csv(DFNINVERSE.obs_data).values
    simulation = DFNINVERSE.read_simulation_results(state).values

    root_mean_square = np.sqrt(np.sum((observation-simulation)**2)/(observation.shape[0]*observation.shape[1]))

    return root_mean_square

def prior_probability():

    return prior

def likelyhood(cov, g, d):

    coeff = 1 /



def parse_parameter_to_input(parameter_table):


    n_fracs = parameter_table.shape[0]

    input_param = {}

    input_param['nUserEll'] = n_fracs
    input_param['Aspect_Ratio'] = '\n'.join(str(e) for e in (np.ones(n_fracs) * 0.6).tolist())
    input_param['N_Vertices'] = '\n'.join(str(e) for e in (np.ones(n_fracs, dtype=int) * 5).tolist())
    input_param['AngleOption'] = '\n'.join(str(e) for e in np.ones(n_fracs, dtype=int).tolist())
    input_param['Beta'] = '\n'.join(str(e) for e in np.zeros(n_fracs).tolist())

    # Parse normal vector
    normal_vectors = np.asarray(
        [(np.cos(parameter_table['psi']) * np.cos(parameter_table['phi'])).tolist(),
         (np.cos(parameter_table['psi']) * np.sin(parameter_table['phi'])).tolist(),
         (np.sin(parameter_table['psi'])).tolist()]
        ).T
    line = []
    for nv in normal_vectors:
        line.append('{' + ', '.join(str(e) for e in nv) + '}')
    nn = '\n'.join(e for e in line)
    input_param['Normal'] = nn

    # Parse radius
    input_param['Radii'] = '\n'.join(str(e) for e in parameter_table['radius'].tolist())

    # Parse center
    centers = np.asarray([parameter_table['center_x'].tolist(),
                          parameter_table['center_y'].tolist(),
                          parameter_table['center_z'].tolist()]
                         ).T
    line = []
    for ct in centers:
        line.append('{' + ', '.join(str(e) for e in ct) + '}')
    cc = '\n'.join(e for e in line)
    input_param['Translation'] = cc

    return input_param



def run_mcmc(variables, max_iter, dodr):

    move_type = ['Shape']

    param_old = variables.initial_state()
    prior_old = variables.get_prior(param_old)
    DFN_input = parse_parameter_to_input(param_old)
    DFNINVERSE.run_forward_simulation(DFN_input, 'old')
    likelihood_old = State.get_likelihood(self.rms('old'))

    count = 0
    while count <= max_iter:

        accept_flag = False
        accept_amount = 0

        movement = move_type[randint(4, size=1)]

        param_new = State.get_proposal(param_old, movement)
        prior_new = State.get_prior(param_new)
        DFNINVERSE.run_forward_simulation(param_new, 'new')
        likelihood_new = State.get_likelihood(self.rms('new'))

        ratio_likelihood = likelihood_new / likelihood_old
        ratio_prior = prior_new / prior_old
        ratio_proposal = State.get_proposal(param_new, movement)

        jacob = State.get_jacob(movement)

        alpha_1 = min(1, ratio_likelihood * ratio_prior * ratio_proposal * jacob)

        beta = random()

        if alpha_1 > beta:
            accept_flag = True
            accept_amount += 1

            DFNINVERSE.swap_states()
            DFNINVERSE.save_accept_model()

            param_old = param_new.copy()
            prior_old = prior_new.copy()
            likelihood_old = likelihood_new.copy()

            State.update()

















