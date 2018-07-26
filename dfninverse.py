import os, shutil, vtk, sys
from string import Template
import pandas as pd
import numpy as np
from subprocess import STDOUT, Popen
from helper import stdout_redirect, define_paths

class DFNINVERSE:

    def __init__(self, project_path, observe_points, domainSize, ncpu=1, **kwargs):

        define_paths()
        flow_condition = kwargs.get('flow_condition',
                                    [['front', 5, 'back', 1],
                                     ['left', 5, 'right', 1],
                                     ['top', 5, 'bottom', 1]])
        relative_meshsize = kwargs.get('relative_meshsize', 100)
        self.project = project_path
        self.ncpu = ncpu
        self.obs_points = observe_points

        # Define directories that carries the job
        self.forward_project_dir = self.project + '/forward_simulation'
        self.accept_model_dir = self.project + '/accept_models'
        self.forward_inputs_dir = self.project + '/input_files'
        self.input_templates_dir = os.environ['PYDFNINV_PATH'] + '/dfnWorks_input_templates'

        # Define functional files
        self.job_report = self.project + '/job_report.txt'
        self.mcmc_log = self.project + "/mcmc_log.txt"
        self.mesh_file_path = self.forward_project_dir + '/full_mesh.vtk'
        self.flow_files_path = self.forward_project_dir + '/PFLOTRAN/parsed_vtk/'
        self.sim_results = self.forward_project_dir + "/forward_results.csv"
        self.h = min(domainSize) / relative_meshsize
        self.domainSize = domainSize
        self.flow_condition = self.__parse_flowcondition(flow_condition)
        self.__make_project_dir()

        open(self.project + '/log_file.log', 'a').close()

    def __parse_flowcondition(self, fc):
        zone_files = ['pboundary_left_w.ex',
                      'pboundary_right_e.ex',
                      'pboundary_front_s.ex',
                      'pboundary_back_n.ex',
                      'pboundary_top.ex',
                      'pboundary_bottom.ex']
        face_names = ['left', 'right', 'front', 'back', 'top', 'bottom']
        flow_cond = []
        for bc in fc:
            flow_cond.append(dict(inflow_region=zone_files[face_names.index(bc[0])],
                                  inflow_pressure=str(bc[1]) + '.d6',
                                  outflow_region=zone_files[face_names.index(bc[2])],
                                  outflow_pressure=str(bc[3]) + '.d6')
                             )
        return flow_cond

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
        if not os.path.isdir(self.accept_model_dir):
            os.mkdir(self.accept_model_dir)
        if not os.path.isdir(self.forward_project_dir):
            os.mkdir(self.forward_project_dir)
        if not os.path.isdir(self.forward_inputs_dir):
            os.mkdir(self.forward_inputs_dir)

        # boundaryFaces = [1 if face in flow_condition else 0 for face in face_names]
        # get dfnTrans input file
        # shutil.copy2(self.template_files + '/PTDFN_control.dat',
        #              self.forward_input_file + '/PTDFN_control.dat')

    def __write_forward_inputs(self, parameters):

        if parameters is None:
            shutil.copy2(self.input_templates_dir + '/define_4_user_ellipses.dat',
                         self.forward_inputs_dir + '/user_define_fractures.dat')
        else:
            params_lists = self.__parse_parameter_to_input(parameters)
            self.write_template(self.input_templates_dir + '/define_user_ellipses.i',
                                self.forward_inputs_dir + '/user_define_fractures.dat',
                                params_lists)
        return None

    def __parse_parameter_to_input(self, parameters):

        parameter_table = pd.DataFrame(parameters,
                                       columns=['center_x', 'center_y', 'center_z',
                                                'phi', 'psi', 'radius', 'aspect_ratio', 'beta', 'n_vertices'])

        # parameter_table = parameter_table.round(5)

        n_fracs = parameter_table.shape[0]

        input_param = {}

        # set number of fractures
        input_param['nUserEll'] = n_fracs
        # set aspect ratio of ellipse
        input_param['Aspect_Ratio'] = '\n'.join(str(e) for e in parameter_table['aspect_ratio'].tolist())
        # set number of vertices
        input_param['N_Vertices'] = '\n'.join(str(int(e)) for e in parameter_table['n_vertices'].tolist())
        # set all angles in degree
        input_param['AngleOption'] = '\n'.join(str(e) for e in np.ones(n_fracs, dtype=int).tolist())
        # set rotation around center (in degree)
        input_param['Beta'] = '\n'.join(str(e) for e in (parameter_table['beta']).tolist())
        # Set radius
        input_param['Radii'] = '\n'.join(str(e) for e in parameter_table['radius'].tolist())


        # Convert normal vector from Sphere coordinate into Cartesian Coordinate
        normal_vectors = np.asarray(
            [(np.cos(parameter_table['psi']) * np.cos(parameter_table['phi'])).tolist(), # x
             (np.cos(parameter_table['psi']) * np.sin(parameter_table['phi'])).tolist(), # y
             (np.sin(parameter_table['psi'])).tolist()] # z
        ).round(3).T
        normal_vectors[np.where(normal_vectors==0)] = 0
        # Set Normal Vector for each ellipse
        line = []
        for nv in normal_vectors:
            line.append('{' + ', '.join(str(e) for e in nv) + '}')
        nn = '\n'.join(e for e in line)
        input_param['Normal'] = nn

        # Set center (translation)
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

    def write_template(self, src, dst, para_list):

        template_file = open(src, 'r').read()
        generate_file = open(dst, 'w+')
        s = Template(template_file)
        generate_file.write(s.safe_substitute(para_list))
        generate_file.close()

        return None

    def __get_observation_ids(self, obs_points):

        eps = self.h * 5
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
            if dist[id] <= eps:
                obs_ids.append(id)
            else:
                obs_ids.append(None)

        # print(obs_ids)
        return obs_ids

    def __get_observation_scalars(self, obs_ids, var_name):
        path, dirs, files = os.walk(self.flow_files_path).__next__()

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

    def read_forward(self, variable_name):
        # with open(self.job_report, "a") as outfile:
        print('\n{0}\nStart Reading Simulation Results.\n{0}\n'.format('=' * 60))
        if os.path.exists(self.mesh_file_path):
            print('Check Mesh file: {} exist.\n'.format(self.mesh_file_path))
            try:
                obs_ids = self.__get_observation_ids(self.obs_points)
                print('Observation Points: \n{}\n'.format(self.obs_points))
                print('Observation Point ID in grid: {}\n'.format(obs_ids))

                obs_scalars = self.__get_observation_scalars(obs_ids, variable_name)

                # Store varialbe value in Dataframe
                df_scalar = pd.DataFrame.from_dict(data=obs_scalars, orient='index', dtype=np.float32)
                df_scalar = df_scalar / 1e6
                i = 1
                columns_name = []
                while i <= len(self.obs_points):
                    columns_name.append('obs_' + str(i))
                    i += 1
                df_scalar.columns = columns_name
                df_scalar.to_csv(self.sim_results)
            except Exception as e:
                print('Get Error when processing data: {}\n'.format(str(e)))
                df_scalar = None
        else:
            print('Check Mesh file: {} exist. Failed!\n'.format(self.mesh_file_path))
            df_scalar = None

        if df_scalar is None:
            print('No Data is read in!\n')
        else:
            print('Read Data Success!\n{}\n'.format('=' * 60))

        return None if df_scalar is None else df_scalar.values

    def run_forward(self, input_parameters, **kwargs):

        variable_name = kwargs.get('variable_name', 'Liquid_Pressure')

        self.__write_forward_inputs(input_parameters)

        jobname = self.forward_project_dir

        # get dfnGen input file and run dfnGen
        self.write_template(self.input_templates_dir + '/gen_user_ellipses.i',
                            self.forward_inputs_dir + '/gen_user_ellipses.dat',
                            {'UserEll_Input_File_Path': self.forward_inputs_dir + '/user_define_fractures.dat',
                             'domainSize': '{' + ', '.join(str(e) for e in self.domainSize) + '}',
                             'h': self.h}
                            )
        for fc in self.flow_condition:
            inflow = fc['inflow_region'].split('_')[1].split('.')[0]
            outflow = fc['outflow_region'].split('_')[1].split('.')[0]
            self.write_template(self.input_templates_dir + '/dfn_explicit.i',
                                self.forward_inputs_dir + '/dfn_explicit_' + inflow + '_' + outflow + '.in',
                                fc
                                )
        run_dfnworks_cmd = ['python3', os.environ['PYDFNINV_PATH'] + '/dfnworks.py',
                            '-j', jobname,
                            '-i', self.forward_inputs_dir,
                            '-n', str(self.ncpu)]
        with open(self.job_report, "a") as outfile:
            p = Popen(run_dfnworks_cmd, stdout=outfile, stderr=STDOUT)
            p.wait()

        with open(self.job_report, "a") as f:
            with stdout_redirect(f):
                syn_data = self.read_forward(variable_name)
        return syn_data

    def gen_3d_obs_points_plot(self, obs_points, radius=0.02):

        forward_project = self.forward_project_dir

        sphere = vtk.vtkSphereSource()
        sphere.SetPhiResolution(21)
        sphere.SetThetaResolution(21)
        sphere.SetRadius(radius)

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
        model_dir = self.accept_model_dir + '/model_' + str(model_id)
        files_to_keep = ['full_mesh.inp', 'full_mesh.vtk', '/forward_results.csv']

        if save_flag:
            os.mkdir(model_dir)
            for file in files_to_keep:
                try:
                    src = self.forward_project_dir + '/' + file
                    if os.path.isdir(src):
                        shutil.copytree(src, model_dir + '/' + file)
                    else:
                        shutil.copy2(src, model_dir)
                except Exception:
                    continue
            shutil.copy2(self.forward_inputs_dir + '/user_define_fractures.dat', model_dir)

        # write mcmc_log
        s = status
        with open(self.mcmc_log, 'a+') as logfile:
            logfile.write('{0}\n{1}\n{2}\n{3}\n'.format('*' * 60, s['dfn_id'], s['rms'], s['dfn']))
