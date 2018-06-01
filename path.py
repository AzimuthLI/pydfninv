import os
import platform as ptf


def valid(name):
    if not (os.path.isfile(os.path.abspath(os.environ[name])) or os.path.isdir(os.path.abspath(os.environ[name]))):
        error_msg = "ERROR: " + name + " has an invalid path name: " + os.environ[name]
        print(error_msg)
        exit()


def define_paths():

    # ================================================
    # THESE PATHS MUST BE SET BY THE USER.
    # ================================================

    if ptf.system() == 'Linux':

        os.environ['PYDFNINV_PATH'] = '/cluster/home/lishi/pydfninv'

        # the dfnWorks-Version2.0  repository
        os.environ['DFNWORKS_PATH'] = '/cluster/project/geg/apps/dfnWorks-Version2.0/'
        valid('DFNWORKS_PATH')
        if not (os.path.isdir(os.path.abspath(os.environ['DFNWORKS_PATH'] + 'tests/'))):
            print("INVALID VERSION OF DFNWORKS - does not have tests folder of official release 2.0")
            exit()

        # PETSC paths
        os.environ['PETSC_DIR'] = '/cluster/project/geg/apps/petsc/petsc-xsdk-0.2.0/gcc-4.8.2'
        os.environ['PETSC_ARCH'] = ''
        valid('PETSC_DIR')

        # PFLOTRAN path
        os.environ['PFLOTRAN_DIR'] = '/cluster/project/geg/apps/pflotran'

        # Python executable
        os.environ['python_dfn'] = '/cluster/apps/python/2.7.14/x86_64/bin/python'
        valid('python_dfn')

        # LaGriT executable
        os.environ['lagrit_dfn'] = '/cluster/project/geg/apps/LaGriT/src/lagrit'
        valid('lagrit_dfn')

    elif ptf.system() == 'Darwin':

        os.environ['PYDFNINV_PATH'] = '/Volumes/SD_Card/Thesis_project/pydfninv'

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
