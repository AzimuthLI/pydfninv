import os, shutil, subprocess, sys, getopt
from time import time

import mesh_dfn_helper as mh
import lagrit_scripts as lagrit
import run_meshing as run_mesh


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


def commandline_parser(argv):
    jobname = ''
    ncpu = ''
    try:
        opts, args = getopt.getopt(argv, "hj:i:n:", ["jobname=", "input=", "ncpu="])
    except getopt.GetoptError:
        print('dfnworks.py -jobname <jobname_path> -input <input_filepath> -ncpu <number_of_CPUs>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('dfnworks.py -jobname <jobname_path> -input <input_filepath> -ncpu <number_of_CPUs>')
            sys.exit()
        elif opt in ("-j", "--jobname"):
            jobname = arg
        elif opt in ("-i", "--input"):
            input_filepath = arg
        elif opt in ("-n", "--ncpu"):
            ncpu = arg

    return jobname, input_filepath, ncpu


if __name__ == '__main__':

    jobname, forward_input_file, ncpu = commandline_parser(sys.argv[1:])

    dfn = DFNWORKS(jobname=jobname, ncpu=int(ncpu))
    dfn._dfnGen_file = forward_input_file + '/gen_user_ellipses.dat'
    dfn._local_dfnGen_file = 'gen_user_ellipses.dat'

    dfn._dfnFlow_file = forward_input_file + '/dfn_explicit.in'
    dfn._local_dfnFlow_file = 'dfn_explicit.in'

    dfn._dfnTrans_file = forward_input_file + '/PTDFN_control.dat'
    dfn._local_dfnTrans_file = 'PTDFN_control.dat'

    dfn._aper_file = 'aperture.dat'
    dfn._perm_file = 'perm.dat'

    # Run forward simulation
    dfn.dfn_gen()
    dfn.dfn_flow()
    dfn.clean_up_files()

