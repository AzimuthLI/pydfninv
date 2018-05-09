import os, shutil, vtk, subprocess, sys
from string import Template
from time import time
import pandas as pd
import numpy as np

class DFNINVERSE:

    os.environ['PYDFNINV_PATH'] = '/Volumes/SD_Card/Thesis_project/pydfninv'

    def __init__(self, project_path, obs_data, ncpu):

        self.project = project_path
        self.obs = obs_data
        self.ncpu = ncpu
        self.__make_project_dir()

    def __make_project_dir(self):

        self.forward_project = self.project + '/forward_simulation'
        self.accept_model_path = self.project + '/accept_models'
        self.forward_input_file = self.project + '/input_files'

        # create project directory
        if os.path.isdir(self.project):
            if not os.listdir(self.project) == []:
                for root, dirs, files in os.walk(self.project):
                    for f in files:
                        os.unlink(os.path.join(root, f))
                    for d in dirs:
                        shutil.rmtree(os.path.join(root, d))
        else:
            os.mkdir(self.project)

        # create directory for the forward simulation, accepted models, input_file
        if not os.path.isdir(self.accept_model_path):
            os.mkdir(self.accept_model_path)
        if not os.path.isdir(self.forward_project):
            os.mkdir(self.forward_project)
        if not os.path.isdir(self.forward_input_file):
            os.mkdir(self.forward_input_file)

        self.template_files = os.environ['PYDFNINV_PATH'] + '/dfnWorks_input_templates'
        self.__write_template_file(self.template_files + '/gen_user_ellipses.i',
                                   self.forward_input_file + '/gen_user_ellipses.dat',
                                   {'UserEll_Input_File_Path':
                                        self.forward_input_file + '/user_define_fractures.dat'}
                                   )
        shutil.copy2(self.template_files + '/dfn_explicit.in',
                     self.forward_input_file + '/dfn_explicit.in')
        shutil.copy2(self.template_files + '/PTDFN_control.dat',
                     self.forward_input_file + '/PTDFN_control.dat')

    def run_forward_simulation(self, input_parameters):

        self.write_forward_inputs(input_parameters)

        # Initialize dfnWorks object
        dfn = DFNWORKS(jobname=self.forward_project, ncpu=self.ncpu)

        dfn._dfnGen_file = self.forward_input_file + '/gen_user_ellipses.dat'
        dfn._local_dfnGen_file = 'gen_user_ellipses.dat'

        dfn._dfnFlow_file = self.forward_input_file + '/dfn_explicit.in'
        dfn._local_dfnFlow_file = 'dfn_explicit.in'

        dfn._dfnTrans_file = self.forward_input_file + '/PTDFN_control.dat'
        dfn._local_dfnTrans_file = 'PTDFN_control.dat'

        dfn._aper_file = 'aperture.dat'
        dfn._perm_file = 'perm.dat'

        # Run forward simulation
        dfn.dfn_gen()
        dfn.dfn_flow()
        dfn.clean_up_files()

        return None

    def write_forward_inputs(self, input_parameters):

        if input_parameters == None:
            shutil.copy2(self.template_files + '/define_4_user_ellipses.dat',
                         self.forward_input_file + '/user_define_fractures.dat')
        else:
            self.__write_template_file(self.template_files + '/define_user_ellipses.i',
                                       self.forward_input_file + '/user_define_fractures.dat',
                                       input_parameters)
        return None

    def __write_template_file(self, src, dst, para_list):

        template_file = open(src, 'r').read()
        generate_file = open(dst, 'w+')
        s = Template(template_file)
        generate_file.write(s.safe_substitute(para_list))
        generate_file.close()

        return None

    def read_simulation_results(self, obs_points, variable_name):

        self.mesh_file_path = self.forward_project + '/full_mesh.vtk'
        self.flow_files_path = self.forward_project + '/PFLOTRAN/parsed_vtk/'

        obs_ids = self.__get_observation_ids(obs_points)
        obs_scalars = self.__get_observation_scalars(obs_ids, variable_name)

        # Store varialbe value in Dataframe and plot
        df_scalar = pd.DataFrame.from_dict(data=obs_scalars, orient='index', dtype=np.float32)
        i = 1
        columns_name = []
        while i <= len(obs_points):
            columns_name.append('obs_' + str(i))
            i += 1

        df_scalar.columns = columns_name
        df_scalar.to_csv(self.forward_project + "forward_simulation_results.csv")

        return df_scalar

    def __get_observation_ids(self, obs_points):

        # Read mesh file (*.vtk)
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(self.mesh_file_path)
        reader.Update()  # Needed because of GetScalarRange
        output = reader.GetOutput()

        # Total Num. of Points in the mesh
        N = output.GetNumberOfPoints()

        # Locate observation point in the mesh
        obs_ids = []
        for obs_pt in obs_points:
            dist = []
            i = 0
            while i < N:
                dist_sq = vtk.vtkMath.Distance2BetweenPoints(obs_pt, output.GetPoint(i))
                dist.append(np.sqrt(dist_sq))
                i += 1
            id = np.argmin(dist)
            obs_ids.append(id)

        return obs_ids

    def __get_observation_scalars(self, obs_ids, var_name):
        path, dirs, files = os.walk(self.flow_files_path).__next__()
        file_count = len(files)
        print("Total time steps: %d \n" % file_count)

        file_num = 1
        obs_scalars = {}
        for vtk_file in files:

            # Read the source file.
            reader = vtk.vtkUnstructuredGridReader()
            reader.SetFileName(path + vtk_file)
            reader.Update()  # Needed because of GetScalarRange
            output = reader.GetOutput()

            key_value = 'time ' + str(file_num)
            obs_scalar = []
            for id in obs_ids:
                obs_scalar.append(output.GetPointData().GetScalars(var_name).GetValue(id))
            obs_scalars.update({key_value: obs_scalar})
            file_num += 1

        return obs_scalars

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
        writer.SetFileName(self.forward_project + '/obs_points.vtk')
        writer.Update()

def valid(name):
    if not (os.path.isfile(os.path.abspath(os.environ[name])) or os.path.isdir(os.path.abspath(os.environ[name]))):
        error_msg = "ERROR: " + name + " has an invalid path name: " + os.environ[name]
        print(error_msg)
        exit()

def define_paths():
    # ================================================
    # THESE PATHS MUST BE SET BY THE USER.
    # ================================================

    # the dfnWorks-Version2.0  repository
    os.environ['DFNWORKS_PATH'] = '/Users/shiyili/projects/dfnWorks/dfnWorks-Version2.0/'
    valid('DFNWORKS_PATH')
    if not (os.path.isdir(os.path.abspath(os.environ['DFNWORKS_PATH'] + 'tests/'))):
        print("INVALID VERSION OF DFNWORKS - does not have tests folder of official release 2.0")
        exit()

    # PETSC paths
    os.environ['PETSC_DIR'] = '/Users/shiyili/projects/dfnWorks/petsc'
    os.environ['PETSC_ARCH'] = 'arch-darwin-c-opt'
    valid('PETSC_DIR')
    #    valid('PETSC_ARCH')

    # PFLOTRAN path
    os.environ['PFLOTRAN_DIR'] = '/Users/shiyili/projects/dfnWorks/pflotran'
    valid('PFLOTRAN_DIR')

    # Python executable
    os.environ['python_dfn'] = '/opt/moose/miniconda/bin/python'
    valid('python_dfn')

    # LaGriT executable
    #    os.environ['lagrit_dfn'] = '/n/swqa/LAGRIT/lagrit.lanl.gov/downloads/lagrit_ulin3.2'
    os.environ['lagrit_dfn'] = '/Users/shiyili/projects/dfnWorks/LaGriT/src/lagrit'
    valid('lagrit_dfn')

    # ===================================================
    # THESE PATHS ARE AUTOMATICALLY SET. DO NOT CHANGE.
    # ====================================================

    # Directories
    os.environ['DFNGEN_PATH'] = os.environ['DFNWORKS_PATH'] + 'DFNGen/'
    os.environ['DFNTRANS_PATH'] = os.environ['DFNWORKS_PATH'] + 'ParticleTracking/'
    os.environ['PYDFNWORKS_PATH'] = os.environ['DFNWORKS_PATH'] + 'pydfnworks/'
    os.environ['connect_test'] = os.environ['DFNWORKS_PATH'] + 'DFN_Mesh_Connectivity_Test/'
    os.environ['correct_uge_PATH'] = os.environ['DFNWORKS_PATH'] + 'C_uge_correct/'
    os.environ['VTK_PATH'] = os.environ['DFNWORKS_PATH'] + 'inp_2_vtk/'

def move_files(file_list, dir_name):
    os.mkdir(dir_name)
    for fle in os.listdir(os.getcwd()):
        for name in file_list:
            if name in fle:
                subprocess.call('mv ' + fle + ' ' + dir_name, shell=True)

def update_progress(i, total, tic):
    progress = i / total
    toc = time()
    dt = toc - tic

    barLength = 20  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength * progress))
    text = "\rPercent: [{0}] {1:.2f}% {2}, {3} / {4} finished. Average time: {5:.5f}" \
        .format("#" * block + "-" * (barLength - block), progress * 100, status, i, total, dt / i)
    sys.stdout.write(text)
    sys.stdout.flush()

import mesh_dfn_helper as mh
import lagrit_scripts as lagrit
import run_meshing as run_mesh

class DFNWORKS:
    from flow import lagrit2pflotran, pflotran, parse_pflotran_vtk_python, pflotran_cleanup, \
        write_perms_and_correct_volumes_areas, zone2ex, parse_pflotran_vtk, inp2vtk_python

    def __init__(self, jobname='', dfnGen_file='', ncpu='', dfnFlow_file='', dfnTrans_file='',
                 inp_file='full_mesh.inp', uge_file='', vtk_file='', mesh_type='dfn',
                 perm_file='', aper_file='', perm_cell_file='', aper_cell_file=''):

        self._jobname = jobname
        self._ncpu = ncpu
        self._local_jobname = self._jobname.split('/')[-1]

        self._dfnGen_file = dfnGen_file
        self._local_dfnGen_file = self._dfnGen_file.split('/')[-1]

        self._output_file = self._dfnGen_file.split('/')[-1]

        self._dfnFlow_file = dfnFlow_file
        self._local_dfnFlow_file = self._dfnFlow_file.split('/')[-1]

        self._dfnTrans_file = dfnTrans_file
        self._local_dfnTrans_file = self._dfnTrans_file.split('/')[-1]

        self._vtk_file = vtk_file
        self._inp_file = inp_file
        self._uge_file = uge_file
        self._mesh_type = mesh_type
        self._perm_file = perm_file
        self._aper_file = aper_file
        self._perm_cell_file = perm_cell_file
        self._aper_cell_file = aper_cell_file

    def dfn_gen(self):

        define_paths()
        # make working directories
        if not os.listdir(self._jobname) == []:
            for root, dirs, files in os.walk(self._jobname):
                for f in files:
                    os.unlink(os.path.join(root, f))
                for d in dirs:
                    shutil.rmtree(os.path.join(root, d))

        shutil.copy2(self._dfnGen_file, self._jobname + '/' + self._local_dfnGen_file)
        os.mkdir(self._jobname + '/radii')
        os.mkdir(self._jobname + '/intersections')
        os.mkdir(self._jobname + '/polys')
        os.chdir(self._jobname)

        tic_gen = time()
        # Create network
        tic = time()
        os.system(os.environ['DFNGEN_PATH'] + '/./DFNGen ' +
                  self._local_dfnGen_file + ' ' + self._jobname)
        run_time_log = open(self._jobname + '/run_time_log.txt', 'w')
        run_time_log.write('Function: dfnGen: ' + str(time() - tic) + ' seconds\n')
        run_time_log.close()

        if os.path.isfile("params.txt") is False:
            print('--> Generation Failed')
            print('--> Exiting Program')
            exit()
        else:
            print('-' * 80)
            print("Generation Succeeded")
            print('-' * 80)

        tic = time()
        self.mesh_network()
        run_time_log = open(self._jobname + '/run_time_log.txt', 'a')
        run_time_log.write('Function: dfn_mesh: ' + str(time() - tic) + ' seconds\n')
        run_time_log.write('Process: dfnGen: ' + str(time() - tic_gen) + ' seconds\n\n')
        run_time_log.close()

    def mesh_network(self, production_mode=True, refine_factor=1, slope=2):
        '''
        Mesh Fracture Network using ncpus and lagrit
        meshing file is separate file: dfnGen_meshing.py
        '''
        print('=' * 80)
        print("Meshing Network Using LaGriT : Starting")
        print('=' * 80)

        num_poly, h, visual_mode, dudded_points, domain = mh.parse_params_file()

        # if number of fractures is greater than number of CPUS,
        # only use num_poly CPUs. This change is only made here, so ncpus
        # is still used in PFLOTRAN
        ncpu = min(self._ncpu, num_poly)
        lagrit.create_parameter_mlgi_file(num_poly, h, slope=slope)
        lagrit.create_lagrit_scripts(visual_mode, ncpu)
        lagrit.create_user_functions()
        failure = run_mesh.mesh_fractures_header(num_poly, ncpu, visual_mode)
        if failure:
            mh.cleanup_dir()
            sys.exit("One or more fractures failed to mesh properly.\nExiting Program")

        n_jobs = lagrit.create_merge_poly_files(ncpu, num_poly, h, visual_mode)

        run_mesh.merge_the_meshes(num_poly, ncpu, n_jobs, visual_mode)

        if not visual_mode:
            if not mh.check_dudded_points(dudded_points):
                mh.cleanup_dir()
                sys.exit("Incorrect Number of dudded points.\nExiting Program")

        if production_mode:
            mh.cleanup_dir()

        if not visual_mode:
            lagrit.define_zones(h, domain)

        mh.output_meshing_report(visual_mode)

    def dfn_flow(self):
        ''' dfnFlow
            Run the dfnFlow portion of the workflow.
            '''

        print('=' * 80)
        print("\ndfnFlow Starting\n")
        print('=' * 80)

        tic_flow = time()

        tic = time()
        self.lagrit2pflotran()
        run_time_log = open(self._jobname + '/run_time_log.txt', 'a')
        run_time_log.write('Function: lagrit2pflotran: ' + str(time() - tic) + ' seconds\n')
        run_time_log.close()

        tic = time()
        self.pflotran()
        run_time_log = open(self._jobname + '/run_time_log.txt', 'a')
        run_time_log.write('Function: pflotran: ' + str(time() - tic) + ' seconds\n')
        run_time_log.close()

        tic = time()
        self.parse_pflotran_vtk_python()
        run_time_log = open(self._jobname + '/run_time_log.txt', 'a')
        run_time_log.write('Function: parse_pflotran_vtk: ' + str(time() - tic) + ' seconds\n')
        run_time_log.close()

        tic = time()
        self.pflotran_cleanup()
        run_time_log = open(self._jobname + '/run_time_log.txt', 'a')
        run_time_log.write('Function: parse_cleanup: ' + str(time() - tic) + ' seconds\n' +
                           'Process: dfnFlow: ' + str(time() - tic_flow) + ' seconds')
        run_time_log.close()

        print('=' * 80)
        print("\ndfnFlow Complete\n")
        print('=' * 80)

    def clean_up_files(self):

        main_list = ['allboundaries.zone', 'aperture.dat', 'cellinfo.dat',
                     'darcyvel.dat', 'dfnTrans_ouput_dir', 'params.txt',
                     'PTDFN_control.dat', 'pboundary', 'zone', 'poly_info.dat',
                     '.inp', 'id_tri_node', 'intersections', 'full_mesh.inp', 'tri_fracture.stor',
                     'cellinfo.dat', 'aperture.dat']
        gen_list = ['DFN_output.txt', 'connectivity.dat', 'families.dat', 'input_generator.dat',
                    'input_generator_clean.dat', 'normal_vectors.dat', 'radii', 'rejections.dat',
                    'rejectsPerAttempt.dat', 'translations.dat', 'triple_points.dat', 'user_rects.dat',
                    'warningFileDFNGen.txt']
        lagrit_list = ['.lgi', 'boundary_output.txt', 'finalmesh.txt',
                       'full_mesh.gmv', 'full_mesh.lg', 'intersections',
                       'lagrit_logs', '3dgen', 'parameters', 'polys']
        pflotran_list = ['dfn_explicit', 'dfn_properties.h5', 'full_mesh.uge',
                         'full_mesh_viz.inp', 'full_mesh_vol_area', 'materialid.dat', 'parsed_vtk', 'perm.dat',
                         'pboundary_', 'convert_uge_params.txt']
        move_files(gen_list, 'DFN_generator')
        move_files(lagrit_list, 'LaGriT')
        move_files(pflotran_list, 'PFLOTRAN')


if __name__ == '__main__':

    dfninv = DFNINVERSE('/Volumes/SD_Card/Thesis_project/dfn_test',
                        '/Volumes/SD_Card/Thesis_project/pydfninv/test_fieldobs.txt', 1)

    input_para_lists = None

    #
    dfninv.run_forward_simulation(input_para_lists)
    #
    obs_points = [(0.4, 0.4, 0.2),
                  (0.4, 0.4, -0.2),
                  (0.4, -0.4, 0.2),
                  (0.4, -0.4, -0.2),
                  (-0.15, -0.08, 0.2)]

    variable_name = ['Liquid_Pressure']

    dfninv.read_simulation_results(obs_points, variable_name[0])

    dfninv.gen_3d_obs_points_plot(obs_points)