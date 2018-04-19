__author__ = "Shiyi Li"
__version__ = "1.0"
__maintainer__ = "Shiyi LI"
__email__ = "lishi@student.ethz.ch"

import os, shutil
from pydfnworks.legal import legal
from pydfnworks import define_paths
import vtk
from dfn_forwards import DFNWORKS

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

    from obs_reading import read_observation

    def __init__(self, iteration=200, project_path=os.getcwd()+'/dfn_inverse', field_obs=' ', ncpu=1):#, prior_setting=' '):

        if os.path.isdir(project_path):
            self._project = project_path
        else:
            os.mkdir(project_path)
            self._project = project_path

        self._iteration = iteration
        self._ncpu = ncpu

        self._field_obs = self._project + '/field_obs.csv'
        # self._prior_setting = self._project + '/prior_setting.dat'
        self._accepted_models = self._project + '/accepted_models/'
        self._forward_model = self._project + '/forward_model/'
        self._data_plots = self._project + '/data_plots/'

        if not os.path.isdir(self._accepted_models):
            os.mkdir(self._accepted_models)
        if not os.path.isdir(self._forward_model):
            os.mkdir(self._forward_model)
        if not os.path.isdir(self._data_plots):
            os.mkdir(self._data_plots)

        if field_obs == ' ':
            if not os.path.exists(self._field_obs):
                raise Exception("Error: No Field Observation is given")
        else:
            if os.path.dirname(field_obs) != self._project:
                shutil.copyfile(field_obs, self._field_obs)
            else:
                os.rename(field_obs, self._field_obs)

        # if prior_setting == ' ':
        #     if not os.path.exists(self._prior_setting):
        #         raise Exception("Error: No Prior setting is given")
        # else:
        #     if os.path.dirname(prior_setting) != self._project:
        #         shutil.copyfile(prior_setting, self._prior_setting)
        #     else:
        #         os.rename(prior_setting, self._prior_setting)

    def run_forward(self):

        input_file_templates = os.getcwd() + '/dfnWorks_input_templates'
        dfn_inputs = {'dfnGen': input_file_templates + '/gen_user_ellipses.dat',
                      'dfnFlow': input_file_templates + '/dfn_explicit.in',
                      'dfnTrans': input_file_templates + '/PTDFN_control.dat',
                      }

        input_file = open(self._forward_model + 'input_dfnWorks.txt', 'w')
        input_file.write('dfnGen ' + dfn_inputs['dfnGen'] + '\n' +
                         'dfnFlow ' + dfn_inputs['dfnFlow'] + '\n' +
                         'dfnTrans ' + dfn_inputs['dfnTrans'])
        input_file.close()

        options = {'jobname': self._forward_model,
                   'ncpu': self._ncpu,
                   'input_file': self._forward_model + 'input_dfnWorks.txt',
                   'cell': False
                   }

        dfn_forward = create_dfn(options)
        dfn_forward.dfn_gen()
        dfn_forward.dfn_flow()
        dfn_forward.cleanup_files_at_end()

    def gen_3d_obs_points_plot(self, obs_points):

        sphere = vtk.vtkSphereSource()
        sphere.SetPhiResolution(21)
        sphere.SetThetaResolution(21)
        sphere.SetRadius(.01)

        filter = vtk.vtkAppendPolyData()

        for pt in obs_points:

            sphere.SetCenter(pt)
            sphere.Update()

            input = vtk.vtkPolyData()
            input.ShallowCopy(sphere.GetOutput())

            filter.AddInputData(input)

        filter.Update()

        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(filter.GetOutput())
        writer.SetFileName(self._forward_model+'/obs_points.vtk')
        writer.Update()


def create_dfn(options):

    '''
    Create dfnworks object
    :param options:
        options = {'jobname': the forward modeling folder path
                    'ncpu': number of cpu cores to run forward modelling
                    'input_file': input file path ('*.txt')
                    'cell': True or False, Binary For Cell Based Apereture / Perm
                    }
    :return: DFNWORKS object
    '''

    dfn = DFNWORKS(jobname=options['jobname'], ncpu=options['ncpu'])
    with open(options['input_file']) as f:
        for line in f:
            line=line.rstrip('\n')
            line=line.split()

            if line[0].find("dfnGen") == 0:
                dfn._dfnGen_file = line[1]
                dfn._local_dfnGen_file = line[1].split('/')[-1]

            elif line[0].find("dfnFlow") == 0:
                dfn._dfnFlow_file = line[1]
                dfn._local_dfnFlow_file = line[1].split('/')[-1]

            elif line[0].find("dfnTrans") == 0:
                dfn._dfnTrans_file = line[1]
                dfn._local_dfnTrans_file = line[1].split('/')[-1]

    if options['cell'] is True:
        dfn._aper_cell_file = 'aper_node.dat'
        dfn._perm_cell_file = 'perm_node.dat'
    else:
        dfn._aper_file = 'aperture.dat'
        dfn._perm_file = 'perm.dat'

    if options['cell'] is True:
        print '--> Expecting Cell Based Aperture and Permeability'
    print("="*80+"\n")

    return dfn



if __name__ == "__main__":
    dfninv = DFN_INVERSE(project_path=os.getcwd()+'/test_creat_dfn', field_obs=os.getcwd() + '/test_fieldobs.txt')#,
                         #prior_setting='/Volumes/SD_Card/Thesis_project/pydfninv/test_priorsetting.txt')

    # dfninv.run_forward()

    # Define observation points and get the id of obs_points in meshgrid
    obs_points = [(0.4, 0.4, 0.2),
                  (0.4, 0.4, -0.2),
                  (0.4, -0.4, 0.2),
                  (0.4, -0.4, -0.2),
                  (-0.15, -0.08, 0.2)]

    variable_name = ['Liquid_Pressure']

    dfninv.read_observation(obs_points, variable_name[0])

    dfninv.gen_3d_obs_points_plot(obs_points)




