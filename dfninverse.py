import os, shutil, vtk
from string import Template
import pandas as pd
import numpy as np
from subprocess import STDOUT, Popen
from path import define_paths
class DFNINVERSE:

    def __init__(self, project_path, observation_data_path, observe_points, ncpu=1):

        define_paths()

        self.project = project_path
        self.forward_project = self.project + '/forward_simulation'
        self.accept_model_path = self.project + '/accept_models'
        self.forward_input_file = self.project + '/input_files'
        self.obs_data = pd.read_csv(observation_data_path, index_col=0).values / 1e6
        self.ncpu = ncpu
        self.obs_points = observe_points
        self.mesh_file_path = self.forward_project + '/full_mesh.vtk'
        self.flow_files_path = self.forward_project + '/PFLOTRAN/parsed_vtk/'
        self.sim_results = self.forward_project + "/forward_results.csv"
        self.mcmc_log = self.project + "/mcmc_log.txt"
        self.__make_project_dir()

    def __make_project_dir(self):

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
                                   {'UserEll_Input_File_Path': self.forward_input_file + '/user_define_fractures.dat'}
                                   )
        shutil.copy2(self.template_files + '/dfn_explicit.in',
                     self.forward_input_file + '/dfn_explicit.in')
        shutil.copy2(self.template_files + '/PTDFN_control.dat',
                     self.forward_input_file + '/PTDFN_control.dat')

    def run_forward(self, input_parameters, **kwargs):

        variable_name = kwargs.get('variable_name', 'Liquid_Pressure')

        self.__write_forward_inputs(input_parameters)

        jobname = self.forward_project

        run_dfnworks_cmd = ['python3', os.environ['PYDFNINV_PATH']+'/dfnworks.py',
                                '-j', jobname,
                                '-i', self.forward_input_file,
                                '-n', str(self.ncpu)]

        dfnworks_job_report = self.project + '/job_report.txt'
        with open(dfnworks_job_report, "w") as outfile:
            p = Popen(run_dfnworks_cmd, stdout=outfile, stderr=STDOUT)
            p.wait()

        syn_data = self.__read_forward(variable_name)

        return syn_data

    def __write_forward_inputs(self, parameters):

        if parameters is None:
            shutil.copy2(self.template_files + '/define_4_user_ellipses.dat',
                         self.forward_input_file + '/user_define_fractures.dat')
        else:
            params_lists = self.__parse_parameter_to_input(parameters)
            self.__write_template_file(self.template_files + '/define_user_ellipses.i',
                                       self.forward_input_file + '/user_define_fractures.dat',
                                       params_lists)
        return None

    def __parse_parameter_to_input(self, parameters):

        parameter_table = pd.DataFrame(parameters,
                                       columns=['center_x', 'center_y', 'center_z', 'phi', 'psi', 'radius'])

        parameter_table = parameter_table.round(1)

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

    def __write_template_file(self, src, dst, para_list):

        template_file = open(src, 'r').read()
        generate_file = open(dst, 'w+')
        s = Template(template_file)
        generate_file.write(s.safe_substitute(para_list))
        generate_file.close()

        return None

    def __read_forward(self, variable_name, save_mode=True):

        if os.path.exists(self.mesh_file_path):

            try:
                obs_ids = self.__get_observation_ids(self.obs_points)
                obs_scalars = self.__get_observation_scalars(obs_ids, variable_name)

                # Store varialbe value in Dataframe and plot
                df_scalar = pd.DataFrame.from_dict(data=obs_scalars, orient='index', dtype=np.float32)
                df_scalar = df_scalar / 1e6
                i = 1
                columns_name = []
                while i <= len(self.obs_points):
                    columns_name.append('obs_' + str(i))
                    i += 1

                df_scalar.columns = columns_name

                if save_mode:
                    df_scalar.to_csv(self.sim_results)

            except Exception:
                df_scalar = None
        else:
            df_scalar = None

        return df_scalar

    def __get_observation_ids(self, obs_points):

        eps = 1e-1
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
                # print(dist_sq)
                dist.append(np.sqrt(dist_sq))
                i += 1
            id = np.argmin(dist)
            if dist[id] <= eps:
                obs_ids.append(id)
            else:
                obs_ids.append(None)
        # print(obs_ids)
        return obs_ids

    def __get_observation_scalars(self, obs_ids, var_name):
        path, dirs, files = os.walk(self.flow_files_path).__next__()
        # file_count = len(files)
        # print("Total time steps: %d \n" % file_count)

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

        forward_project = self.forward_project

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
        writer.SetFileName(forward_project + '/obs_points.vtk')
        writer.Update()

    def write_inverselog(self, status, model_id, save_flag):
        model_dir = self.accept_model_path + '/model_' + str(model_id)
        files_to_keep = ['full_mesh.inp', 'full_mesh.vtk', '/PFLOTRAN/parsed_vtk/', '/forward_results.csv']

        if save_flag:
            os.mkdir(model_dir)
            for file in files_to_keep:
                try:
                    src = self.forward_project + '/' + file
                    if os.path.isdir(src):
                        shutil.copytree(src, model_dir+'/'+file)
                    else:
                        shutil.copy2(src, model_dir)
                except Exception:
                    continue
            shutil.copy2(self.forward_input_file + '/user_define_fractures.dat', model_dir)

        # write mcmc_log
        s = status
        with open(self.mcmc_log, 'a+') as logfile:
            logfile.write('{0}\n{1}\n{2}\n{3}\n'.format('*'*60, s['dfn_id'], s['rms'], s['dfn']))






# if __name__ == '__main__':
#
#     obs_points = [(0.4, 0.4, 0.2),
#                   (0.4, 0.4, -0.2),
#                   (0.4, -0.4, 0.2),
#                   (0.4, -0.4, -0.2),
#                   (-0.15, -0.08, 0.2)]
#
#     dfninv = DFNINVERSE('/Volumes/SD_Card/Thesis_project/dfn_test',
#                         '/Volumes/SD_Card/Thesis_project/pydfninv/test_fieldobs.txt', obs_points)
#
#     obs_frac = {'obs_1': [-0.4, 0, 0, 0, np.pi/2, np.nan],
#                 'obs_2': [0, 0, 0, 0, 0, np.nan],
#                 'obs_3': [0.6, 0, 0.2, 0, np.pi/2, np.nan],
#                 'obs_4': [0.4, 0, -0.2, 0, np.pi/2., np.nan]}

    # s = State(obs_frac)
    #
    # para_list = s.get_initial()
    #
    # input_para_lists = para_list
    #
    # dfninv.run_forward_simulation(input_para_lists, 'new')
    #
    # variable_name = ['Liquid_Pressure']
    #
    # dfninv.read_simulation_results('new')
    #
    # dfninv.gen_3d_obs_points_plot(obs_points, 'new')
    #
    # dfninv.swap_states()
    #
    # dfninv.save_accepted_model(1)