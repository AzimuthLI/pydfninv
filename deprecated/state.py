import numpy as np
import pandas as pd
from numpy.random import randint, uniform, normal, random
from string import Template

class State:

    def __init__(self, observed_fracture, n_range=15, **kwargs):

        self.r_range = kwargs.get('r_range', [0.5, 1])
        self.cx_range = kwargs.get('cx_range', [-0.5, 0.5])
        self.cy_range = kwargs.get('cy_range', [-0.5, 0.5])
        self.cz_range = kwargs.get('cz_range', [-0.5, 0.5])
        self.angle_range = kwargs.get('angle_range', [0, np.pi])

        self.std = kwargs.get('std', 0.1*np.ones(6))

        try:
            self.n_unknownfrac = randint(n_range[0], n_range[1])
        except TypeError:
            self.n_unknownfrac = randint(1, n_range)

        self.n_observed = len(observed_fracture)

        self.parameters_initial = \
            pd.DataFrame.from_dict(observed_fracture,
                                   orient='index',
                                   columns=['center_x', 'center_y', 'center_z', 'phi', 'psi', 'radius'])
        self.parameters_initial.index.name = 'Fracture_ID'


    def get_initial(self):

        self.parameters_initial.loc[0:self.n_observed, 'radius'] = \
            uniform(self.r_range[0], self.r_range[1], self.n_observed)
        #
        unknown_fractures = pd.DataFrame(index=np.arange(self.n_unknownfrac),
                                         columns=['center_x',
                                                  'center_y',
                                                  'center_z',
                                                  'phi', 'psi', 'radius'])

        unknown_fractures['center_x'] = uniform(self.cx_range[0], self.cx_range[1], self.n_unknownfrac)
        unknown_fractures['center_y'] = uniform(self.cy_range[0], self.cy_range[1], self.n_unknownfrac)
        unknown_fractures['center_z'] = uniform(self.cz_range[0], self.cz_range[1], self.n_unknownfrac)
        unknown_fractures['phi'] = uniform(self.angle_range[0], self.angle_range[1], self.n_unknownfrac)
        unknown_fractures['psi'] = uniform(self.angle_range[0], self.angle_range[1], self.n_unknownfrac)
        unknown_fractures['radius'] = uniform(self.r_range[0], self.r_range[1], self.n_unknownfrac)

        self.parameters_initial = self.parameters_initial.append(unknown_fractures)
        self.parameters_initial = self.parameters_initial.round(1)

        return parse_parameter_to_input(self.parameters_initial)

    def get_proposal(self, movement_type):

        self.parameters_proposal = self.parameters_initial.copy()

        if movement_type == 'Shape':

            row_id = randint(self.parameters_proposal.shape[0])
            frac_id = self.parameters_proposal.index[row_id]

            if row_id < self.n_observed:
                var_id = 5
                col = 'radius'
            else:
                var_id = randint(6)
                col = self.parameters_proposal.columns[var_id]

            self.parameters_proposal.loc[frac_id, col] += self.std[var_id] * normal()
            self.parameters_proposal = self.parameters_proposal.round(1)

        elif movement_type == 'Birth':
            pass

        elif movement_type == 'Death':
            pass

        return parse_parameter_to_input(self.parameters_proposal)

    def update(self):

        self.parameters_initial = self.parameters_proposal.copy()

        return None














def write_template(src, dst, para_list):
    template_file = open(src, 'r').read()
    generate_file = open(dst, 'w+')
    s = Template(template_file)
    generate_file.write(s.safe_substitute(para_list))
    generate_file.close()

    return None





if __name__ == '__main__':

    obs_frac = {'obs_1': [0.2, 5, 3, 1.1, 2.6, np.nan],
                'obs_2': [2, 5, 3, 1.1, 2.6, np.nan],
                'obs_3': [2, 5, 3, 1.1, 2.6, np.nan]}

    s = State(obs_frac)

    print(s.get_initial())

    s.get_proposal('Shape')
    # write_template('./dfnWorks_input_templates/define_user_ellipses.i',
    #                './dfnWorks_input_templates/write_input_test.dat', para_list)


    #
    # print(para_list)
