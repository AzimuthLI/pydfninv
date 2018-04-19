__author__ = "Shiyi Li"
__version__ = "1.0"
__maintainer__ = "Shiyi Li"
__email__ = "lishi@student.ethz.ch"

import subprocess
import os, shutil
from time import time
from pydfnworks import define_paths

def move_files(file_list, dir_name):
    os.mkdir(dir_name)
    for fle in os.listdir(os.getcwd()):
        for name in file_list:
            if name in fle:
                subprocess.call('mv ' + fle + ' ' + dir_name, shell=True)

class DFNWORKS:
    """  Class for DFN Generation and meshing

    Attributes:
        * _jobname: name of job, also the folder where output files are stored
        * _ncpu: number of CPUs used in the job
        * _dfnGen file: the name of the dfnGen input file
        * _dfnFlow file: the name of the dfnFlow input file
        * _local prefix: indicates that the name contains only the most local directory
        * _vtk_file: the name of the VTK file
        * _inp_file: the name of the INP file
        * _uge_file: the name of the UGE file
        * _mesh_type: the type of mesh
        * _perm_file: the name of the file containing permeabilities
        * _aper_file: the name of the file containing apertures
        * _perm_cell file: the name of the file containing cell permeabilities
        * _aper_cell_file: the name of the file containing cell apertures
        * _dfnTrans_version: the version of dfnTrans to use
        * _freeze: indicates whether the class attributes can be modified
        * _large_network: indicates whether C++ or Python is used for file processing at the bottleneck
        of inp to vtk conversion
    """

    from meshdfn import mesh_network
    from flow import dfn_flow, lagrit2pflotran, pflotran, parse_pflotran_vtk, inp2vtk_python, parse_pflotran_vtk_python, pflotran_cleanup, write_perms_and_correct_volumes_areas, zone2ex

    def __init__(self, jobname='',dfnGen_file='', ncpu='', dfnFlow_file = '', dfnTrans_file = '', inp_file='full_mesh.inp', uge_file='', vtk_file='', mesh_type='dfn', perm_file='', aper_file='',perm_cell_file='',aper_cell_file=''):

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

        shutil.copy2(self._dfnGen_file, self._jobname+'/'+self._local_dfnGen_file)
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
        run_time_log.write('Function: dfnGen: '+ str(time()-tic) + ' seconds\n')
        run_time_log.close()

        if os.path.isfile("params.txt") is False:
            print '--> Generation Failed'
            print '--> Exiting Program'
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

    def cleanup_files_at_end(self):

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





