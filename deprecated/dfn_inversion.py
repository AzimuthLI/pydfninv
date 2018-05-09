__author__ = "Shiyi Li"
__version__ = "1.0"
__maintainer__ = "Shiyi LI"
__email__ = "lishi@student.ethz.ch"

import os
from deprecated.dfn_forwards import create_dfn

class DFN_INVERSE:
    """
    Class for DFN inversion

    Attributes:
         _project: the full path where the project is stored
         _field_obs: path of the field observation. If the file is not in the project folder, a copy will be generated
                     inside the project folder.
         _prior_setting: path of the prior_setting file. If the file is not in the project folder, a copy will be
                         generated inside the project folder
         _accepted_models: path to save the accepted models, "project/accepted_models/"
         _forward_modeling: path to carry out forward simulation, "project/forward_modeling/"
         _iteration: iterate steps of inversion

    """

    from deprecated.fracture_generator import fracture_update
    from deprecated.helper import write_forward_simulation_options, make_direcories

    os.environ['PYDFNINV_PATH'] = '/Volumes/SD_Card/Thesis_project/pydfninv/'

    def __init__(self, iteration=200, project_path=os.getcwd()+'/dfn_inverse', field_obs=' ', ncpu=1):

        self._project = project_path
        self._iteration = iteration
        self._ncpu = ncpu
        self._field_obs = self._project + '/field_obs.csv'
        self._accepted_models = self._project + '/accepted_models/'
        self._forward_model = self._project + '/forward_model/'
        self._data_plots = self._project + '/data_plots/'
        self.make_direcories(field_obs)

    def run_forward(self, input_para_lists):

        options = self.write_forward_simulation_options(input_para_lists)
        dfn_forward = create_dfn(options)
        dfn_forward.dfn_gen()
        dfn_forward.dfn_flow()
        dfn_forward.cleanup_files_at_end()

if __name__ == "__main__":
    dfninv = DFN_INVERSE(project_path='/Volumes/SD_Card/Thesis_project/inv_test', field_obs=os.getcwd() + '/test_fieldobs.txt')#,
                         #prior_setting='/Volumes/SD_Card/Thesis_project/pydfninv/test_priorsetting.txt')

    fix_fractures = {'centers': '{-0.4,0,0} \n{0,0,-0.32} \n{0.3,0,0.2} \n{0.4,0,-0.2} \n{0.1,0,0.2}',
                     'normal_vectors': '{0,0,1} \n{0.98,0,0.21} \n{0,0,1} \n{0,0,1} \n{0.97,0,0.26}'}

    radii_stats = {'Distribution': 'Normal',
                   'mu': 0.7,
                   'std': 0.1
                   }

    aspect_ratio_stats = {'Distribution': 'Normal',
                          'mu': 0.7,
                          'std': 0.1
                         }

    user_define_para_list = dfninv.fracture_update(fix_fractures)

    input_para_lists = {'dfnGen': {'UserEll_Input_File_Path':
                                       dfninv._project+'/dfnWorks_input_files/define_user_ellipses.dat'},
                        'dfnFlow': None,
                        'dfnTrans': None,
                        'user_define': user_define_para_list}

    dfninv.run_forward(input_para_lists)


    #
    # # Define observation points and get the id of obs_points in meshgrid
    # obs_points = [(0.4, 0.4, 0.2),
    #               (0.4, 0.4, -0.2),
    #               (0.4, -0.4, 0.2),
    #               (0.4, -0.4, -0.2),
    #               (-0.15, -0.08, 0.2)]
    #
    # variable_name = ['Liquid_Pressure']
    #
    # dfninv.read_observation(obs_points, variable_name[0])
    #
    # dfninv.gen_3d_obs_points_plot(obs_points)




