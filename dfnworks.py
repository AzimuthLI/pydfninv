import os, shutil, subprocess, sys, getopt, glob, h5py, re
from time import time
from helper import define_paths
import mesh_dfn_helper as mh
import lagrit_scripts as lagrit
import run_meshing as run_mesh
import numpy as np
import platform as ptf


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


def move_files(file_list, dir_name):
    os.mkdir(dir_name)
    for fle in os.listdir(os.getcwd()):
        for name in file_list:
            if name in fle:
                subprocess.call('mv ' + fle + ' ' + dir_name, shell=True)


class DFNWORKS:
    # from flow import lagrit2pflotran, pflotran, parse_pflotran_vtk_python, pflotran_cleanup, \
    #     write_perms_and_correct_volumes_areas, zone2ex, parse_pflotran_vtk, inp2vtk_python

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

    def lagrit2pflotran(self, inp_file='', mesh_type='', hex2tet=False):
        """  Takes output from LaGriT and processes it for use in PFLOTRAN.

        Kwargs:
            * inp_file (str): name of the inp (AVS) file produced by LaGriT
            * mesh_type (str): the type of mesh
            * hex2tet (boolean): True if hex mesh elements should be converted to tet elements, False otherwise.
        """
        print('=' * 80)
        print("Starting conversion of files for PFLOTRAN ")
        print('=' * 80)
        if inp_file:
            self._inp_file = inp_file
        else:
            inp_file = self._inp_file

        if inp_file == '':
            sys.exit('ERROR: Please provide inp filename!')

        if mesh_type:
            if mesh_type in mesh_types_allowed:
                self._mesh_type = mesh_type
            else:
                sys.exit('ERROR: Unknown mesh type. Select one of dfn, volume or mixed!')
        else:
            mesh_type = self._mesh_type

        if mesh_type == '':
            sys.exit('ERROR: Please provide mesh type!')

        self._uge_file = inp_file[:-4] + '.uge'
        # Check if UGE file was created by LaGriT, if it does not exists, exit
        failure = os.path.isfile(self._uge_file)
        if failure == False:
            sys.exit('Failed to run LaGrit to get initial .uge file')

        if mesh_type == 'dfn':
            self.write_perms_and_correct_volumes_areas()  # Make sure perm and aper files are specified

        # Convert zone files to ex format
        # self.zone2ex(zone_file='boundary_back_n.zone',face='north')
        # self.zone2ex(zone_file='boundary_front_s.zone',face='south')
        # self.zone2ex(zone_file='boundary_left_w.zone',face='west')
        # self.zone2ex(zone_file='boundary_right_e.zone',face='east')
        # self.zone2ex(zone_file='boundary_top.zone',face='top')
        # self.zone2ex(zone_file='boundary_bottom.zone',face='bottom')
        self.zone2ex(zone_file='all')
        print('=' * 80)
        print("Conversion of files for PFLOTRAN complete")
        print('=' * 80)
        print("\n\n")

    def zone2ex(self, uge_file='', zone_file='', face=''):
        '''zone2ex
        Convert zone files from LaGriT into ex format for LaGriT
        inputs:
        uge_file: name of uge file
        zone_file: name of zone file
        face: face of the plane corresponding to the zone file

        zone_file='all' processes all directions, top, bottom, left, right, front, back
        '''

        print('--> Converting zone files to ex')
        if self._uge_file:
            uge_file = self._uge_file
        else:
            self._uge_file = uge_file

        uge_file = self._uge_file
        if uge_file == '':
            sys.exit('ERROR: Please provide uge filename!')
        # Opening uge file
        print('\n--> Opening uge file')
        fuge = open(uge_file, 'r')

        # Reading cell ids, cells centers and cell volumes
        line = fuge.readline()
        line = line.split()
        NumCells = int(line[1])

        Cell_id = np.zeros(NumCells, 'int')
        Cell_coord = np.zeros((NumCells, 3), 'float')
        Cell_vol = np.zeros(NumCells, 'float')

        for cells in range(NumCells):
            line = fuge.readline()
            line = line.split()
            Cell_id[cells] = int(line.pop(0))
            line = [float(id) for id in line]
            Cell_vol[cells] = line.pop(3)
            Cell_coord[cells] = line
        fuge.close()

        print('--> Finished with uge file\n')

        # loop through zone files
        if zone_file is 'all':
            zone_files = ['pboundary_front_s.zone', 'pboundary_back_n.zone', 'pboundary_left_w.zone', \
                          'pboundary_right_e.zone', 'pboundary_top.zone', 'pboundary_bottom.zone']
            face_names = ['south', 'north', 'west', 'east', 'top', 'bottom']
        else:
            if zone_file == '':
                sys.exit('ERROR: Please provide boundary zone filename!')
            if face == '':
                sys.exit('ERROR: Please provide face name among: top, bottom, north, south, east, west !')
            zone_files = [zone_file]
            face_names = [face]

        for iface, zone_file in enumerate(zone_files):
            face = face_names[iface]
            # Ex filename
            ex_file = zone_file.strip('zone') + 'ex'

            # Opening the input file
            print('--> Opening zone file: ', zone_file)
            fzone = open(zone_file, 'r')
            fzone.readline()
            fzone.readline()
            fzone.readline()

            # Read number of boundary nodes
            print('--> Calculating number of nodes')
            NumNodes = int(fzone.readline())
            Node_array = np.zeros(NumNodes, 'int')
            # Read the boundary node ids
            print('--> Reading boundary node ids')

            if (NumNodes < 10):
                g = fzone.readline()
                node_array = g.split()
                # Convert string to integer array
                node_array = [int(id) for id in node_array]
                Node_array = np.asarray(node_array)
            else:
                for i in range(int(NumNodes / 10 + 1)):
                    g = fzone.readline()
                    node_array = g.split()
                    # Convert string to integer array
                    node_array = [int(id) for id in node_array]
                    if (NumNodes - 10 * i < 10):
                        for j in range(NumNodes % 10):
                            Node_array[i * 10 + j] = node_array[j]
                    else:
                        for j in range(10):
                            Node_array[i * 10 + j] = node_array[j]
            fzone.close()
            print('--> Finished with zone file')

            Boundary_cell_area = np.zeros(NumNodes, 'float')
            for i in range(NumNodes):
                Boundary_cell_area[i] = 1.e20  # Fix the area to a large number

            print('--> Finished calculating boundary connections')

            boundary_cell_coord = [Cell_coord[Cell_id[i - 1] - 1] for i in Node_array]
            epsilon = 1e-0  # Make distance really small
            if (face == 'top'):
                boundary_cell_coord = [[cell[0], cell[1], cell[2] + epsilon] for cell in boundary_cell_coord]
            elif (face == 'bottom'):
                boundary_cell_coord = [[cell[0], cell[1], cell[2] - epsilon] for cell in boundary_cell_coord]
            elif (face == 'north'):
                boundary_cell_coord = [[cell[0], cell[1] + epsilon, cell[2]] for cell in boundary_cell_coord]
            elif (face == 'south'):
                boundary_cell_coord = [[cell[0], cell[1] - epsilon, cell[2]] for cell in boundary_cell_coord]
            elif (face == 'east'):
                boundary_cell_coord = [[cell[0] + epsilon, cell[1], cell[2]] for cell in boundary_cell_coord]
            elif (face == 'west'):
                boundary_cell_coord = [[cell[0] - epsilon, cell[1], cell[2]] for cell in boundary_cell_coord]
            elif (face == 'none'):
                boundary_cell_coord = [[cell[0], cell[1], cell[2]] for cell in boundary_cell_coord]
            else:
                sys.exit('ERROR: unknown face. Select one of: top, bottom, east, west, north, south.')

            with open(ex_file, 'w') as f:
                f.write('CONNECTIONS\t%i\n' % Node_array.size)
                for idx, cell in enumerate(boundary_cell_coord):
                    f.write('%i\t%.6e\t%.6e\t%.6e\t%.6e\n' % (
                        Node_array[idx], cell[0], cell[1], cell[2], Boundary_cell_area[idx]))
            print('--> Finished writing ex file "' + ex_file + '" corresponding to the zone file: ' + zone_file + '\n')

        print('--> Converting zone files to ex complete')

    def inp2gmv(self, inp_file=''):
        """ Convert inp file to gmv file, for general mesh viewer .

        Kwargs:
            inp_file (str): name of inp file
        """

        if inp_file:
            self._inp_file = inp_file
        else:
            inp_file = self._inp_file

        if inp_file == '':
            sys.exit('ERROR: inp file must be specified in inp2gmv!')

        gmv_file = inp_file[:-4] + '.gmv'

        with open('inp2gmv.lgi', 'w') as fid:
            fid.write('read / avs / ' + inp_file + ' / mo\n')
            fid.write('dump / gmv / ' + gmv_file + ' / mo\n')
            fid.write('finish \n\n')

        cmd = lagrit_path + ' <inp2gmv.lgi ' + '>lagrit_inp2gmv.txt'
        failure = os.system(cmd)
        if failure:
            sys.exit('ERROR: Failed to run LaGrit to get gmv from inp file!')
        print("--> Finished writing gmv format from avs format")

    def write_perms_and_correct_volumes_areas(self, inp_file='', uge_file='', perm_file='', aper_file=''):
        """ Write permeability values to perm_file, write aperture values to aper_file, and correct volume areas in uge_file
        """
        print("--> Writing Perms and Correct Volume Areas")
        if inp_file:
            self._inp_file = inp_file
        else:
            inp_file = self._inp_file

        if inp_file == '':
            sys.exit('ERROR: inp file must be specified!')

        if uge_file:
            self._uge_file = uge_file
        else:
            uge_file = self._uge_file

        if uge_file == '':
            sys.exit('ERROR: uge file must be specified!')

        if perm_file:
            self._perm_file = perm_file
        else:
            perm_file = self._perm_file

        if perm_file == '' and self._perm_cell_file == '':
            sys.exit('ERROR: perm file must be specified!')

        if aper_file:
            self._aper_file = aper_file
        else:
            aper_file = self._aper_file

        if aper_file == '' and self._aper_cell_file == '':
            sys.exit('ERROR: aperture file must be specified!')

        mat_file = 'materialid.dat'
        t = time()
        # Make input file for C UGE converter
        f = open("convert_uge_params.txt", "w")
        f.write("%s\n" % inp_file)
        f.write("%s\n" % mat_file)
        f.write("%s\n" % uge_file)
        f.write("%s" % (uge_file[:-4] + '_vol_area.uge\n'))
        if self._aper_cell_file:
            f.write("%s\n" % self._aper_cell_file)
            f.write("1\n")
        else:
            f.write("%s\n" % self._aper_file)
            f.write("-1\n")
        f.close()

        cmd = os.environ['correct_uge_PATH'] + 'correct_uge' + ' convert_uge_params.txt'
        failure = os.system(cmd)
        if failure > 0:
            sys.exit('ERROR: UGE conversion failed\nExiting Program')
        elapsed = time() - t
        print('--> Time elapsed for UGE file conversion: %0.3f seconds\n' % elapsed)

        # need number of nodes and mat ID file
        print('--> Writing HDF5 File')
        materialid = np.genfromtxt(mat_file, skip_header=3).astype(int)
        materialid = -1 * materialid - 6
        NumIntNodes = len(materialid)

        if perm_file:
            filename = 'dfn_properties.h5'
            h5file = h5py.File(filename, mode='w')
            print('--> Beginning writing to HDF5 file')
            print('--> Allocating cell index array')
            iarray = np.zeros(NumIntNodes, '=i4')
            print('--> Writing cell indices')
            # add cell ids to file
            for i in range(NumIntNodes):
                iarray[i] = i + 1
            dataset_name = 'Cell Ids'
            h5dset = h5file.create_dataset(dataset_name, data=iarray)

            print('--> Allocating permeability array')
            perm = np.zeros(NumIntNodes, '=f8')

            print('--> reading permeability data')
            print('--> Note: this script assumes isotropic permeability')
            perm_list = np.genfromtxt(perm_file, skip_header=1)
            perm_list = np.delete(perm_list, np.s_[1:5], 1)

            matid_index = -1 * materialid - 7
            for i in range(NumIntNodes):
                j = matid_index[i]
                if int(perm_list[j, 0]) == materialid[i]:
                    perm[i] = perm_list[j, 1]
                else:
                    sys.exit('Indexing Error in Perm File')

            dataset_name = 'Permeability'
            h5dset = h5file.create_dataset(dataset_name, data=perm)

            h5file.close()
            print("--> Done writing permeability to h5 file")
            del perm_list

        if self._perm_cell_file:
            filename = 'dfn_properties.h5'
            h5file = h5py.File(filename, mode='w')

            print('--> Beginning writing to HDF5 file')
            print('--> Allocating cell index array')
            iarray = np.zeros(NumIntNodes, '=i4')
            print('--> Writing cell indices')
            # add cell ids to file
            for i in range(NumIntNodes):
                iarray[i] = i + 1
            dataset_name = 'Cell Ids'
            h5dset = h5file.create_dataset(dataset_name, data=iarray)
            print('--> Allocating permeability array')
            perm = np.zeros(NumIntNodes, '=f8')
            print('--> reading permeability data')
            print('--> Note: this script assumes isotropic permeability')
            f = open(self._perm_cell_file, 'r')
            f.readline()
            perm_list = []
            while True:
                h = f.readline()
                h = h.split()
                if h == []:
                    break
                h.pop(0)
                perm_list.append(h)

            perm_list = [float(perm[0]) for perm in perm_list]

            dataset_name = 'Permeability'
            h5dset = h5file.create_dataset(dataset_name, data=perm_list)
            f.close()

            h5file.close()
            print('--> Done writing permeability to h5 file')

    def pflotran(self):
        ''' Run pflotran
        Copy PFLOTRAN run file into working directory and run with ncpus
        '''
        try:
            shutil.copy(os.path.abspath(self._dfnFlow_file), os.path.abspath(os.getcwd()))
        except:
            print("-->ERROR copying PFLOTRAN input file")
            exit()
        print("=" * 80)
        print("--> Running PFLOTRAN")
        if ptf.system() == 'Linux':
            cmd = '/cluster/apps/openmpi/1.6.5/x86_64/gcc_4.8.2/bin/mpirun -np ' + str(self._ncpu) + \
                  ' ' + os.environ['PFLOTRAN_DIR'] + '/src/pflotran/pflotran -pflotranin ' + self._local_dfnFlow_file
        elif ptf.system() == 'Darwin':
            cmd = os.environ['PETSC_DIR'] + '/' + os.environ['PETSC_ARCH'] + '/bin/mpirun -np ' + str(self._ncpu) + \
                  ' ' + os.environ['PFLOTRAN_DIR'] + '/src/pflotran/pflotran -pflotranin ' + self._local_dfnFlow_file
        os.system(cmd)
        print('=' * 80)
        print("--> Running PFLOTRAN Complete")
        print('=' * 80)
        print("\n")

    def pflotran_cleanup(self):
        '''pflotran_cleanup
        Concatenate PFLOTRAN output files and then delete them
        '''
        print('--> Processing PFLOTRAN output')

        cmd = 'cat ' + self._local_dfnFlow_file[:-3] + '-cellinfo-001-rank*.dat > cellinfo.dat'
        os.system(cmd)

        cmd = 'cat ' + self._local_dfnFlow_file[:-3] + '-darcyvel-001-rank*.dat > darcyvel.dat'
        os.system(cmd)

        for fl in glob.glob(self._local_dfnFlow_file[:-3] + '-cellinfo*.dat'):
            os.remove(fl)
        for fl in glob.glob(self._local_dfnFlow_file[:-3] + '-darcyvel*.dat'):
            os.remove(fl)

        # pflotran_list = ['dfn_explicit', 'dfn_properties.h5', 'full_mesh.uge',
        #                  'full_mesh_viz.inp', 'full_mesh_vol_area', 'materialid.dat', 'parsed_vtk', 'perm.dat',
        #                  'pboundary_', 'convert_uge_params.txt']
        # print('PFLOTRAN result is saved in {}'.format('PFLOTRAN_'+dir))
        # move_files(['parsed_vtk'], 'PFLOTRAN_'+dir)

    def create_dfn_flow_links():
        os.symlink('../full_mesh.uge', 'full_mesh.uge')
        os.symlink('../full_mesh_vol_area.uge', 'full_mesh_vol_area.uge')
        os.symlink('../full_mesh.inp', 'full_mesh.inp')
        os.symlink('../pboundary_back_n.zone', 'pboundary_back_n.zone')
        os.symlink('../pboundary_front_s.zone', 'pboundary_front_s.zone')
        os.symlink('../pboundary_left_w.zone', 'pboundary_left_w.zone')
        os.symlink('../pboundary_right_e.zone', 'pboundary_right_e.zone')
        os.symlink('../pboundary_top.zone', 'pboundary_top.zone')
        os.symlink('../pboundary_bottom.zone', 'pboundary_bottom.zone')
        os.symlink('../materialid.dat', 'materialid.dat')

    def uncorrelated(sigma):
        print('--> Creating Uncorrelated Transmissivity Fields')
        print('Variance: ', sigma)
        print('Running un-correlated')
        x = np.genfromtxt('../aperture.dat', skip_header=1)[:, -1]
        k = np.genfromtxt('../perm.dat', skip_header=1)[0, -1]
        n = len(x)

        print(np.mean(x))

        perm = np.log(k) * np.ones(n)
        perturbation = np.random.normal(0.0, 1.0, n)
        perm = np.exp(perm + np.sqrt(sigma) * perturbation)

        aper = np.sqrt((12.0 * perm))
        aper -= np.mean(aper)
        aper += np.mean(x)

        print('\nPerm Stats')
        print('\tMean:', np.mean(perm))
        print('\tMean:', np.mean(np.log(perm)))
        print('\tVariance:', np.var(np.log(perm)))
        print('\tMinimum:', min(perm))
        print('\tMaximum:', max(perm))
        print('\tMinimum:', min(np.log(perm)))
        print('\tMaximum:', max(np.log(perm)))

        print('\nAperture Stats')
        print('\tMean:', np.mean(aper))
        print('\tVariance:', np.var(aper))
        print('\tMinimum:', min(aper))
        print('\tMaximum:', max(aper))

        output_filename = 'aperture_' + str(sigma) + '.dat'
        f = open(output_filename, 'w+')
        f.write('aperture\n')
        for i in range(n):
            f.write('-%d 0 0 %0.5e\n' % (i + 7, aper[i]))
        f.close()

        cmd = 'ln -s ' + output_filename + ' aperture.dat '
        os.system(cmd)

        output_filename = 'perm_' + str(sigma) + '.dat'
        f = open(output_filename, 'w+')
        f.write('permeability\n')
        for i in range(n):
            f.write('-%d 0 0 %0.5e %0.5e %0.5e\n' % (i + 7, perm[i], perm[i], perm[i]))
        f.close()

        cmd = 'ln -s ' + output_filename + ' perm.dat '
        os.system(cmd)

    def parse_pflotran_vtk(self, grid_vtk_file=''):
        """ Using C++ VTK library, convert inp file to VTK file, then change name of CELL_DATA to POINT_DATA.
        """
        print('--> Parsing PFLOTRAN output using C++')
        files = glob.glob('*-[0-9][0-9][0-9].vtk')
        out_dir = 'parsed_vtk'
        vtk_filename_list = []
        replacements = {'CELL_DATA': 'POINT_DATA'}
        header = ['# vtk DataFile Version 2.0\n',
                  'PFLOTRAN output\n',
                  'ASCII\n']

        inp_file = self._inp_file
        inp_file_copy = self._inp_file[:-4] + '_copy.inp'
        subprocess.call('cp ' + inp_file + ' ' + inp_file_copy, shell=True)
        jobname = self._jobname + '/'

        for fle in files:

            if os.stat(fle).st_size == 0:
                print('ERROR: opening an empty pflotran output file')
                exit()

            temp_file = fle[:-4] + '_temp.vtk'
            with open(fle, 'r') as infile, open(temp_file, 'w') as outfile:
                ct = 0
                for line in infile:
                    if 'CELL_DATA' in line:
                        num_cells = line.strip(' ').split()[1]
                        outfile.write('POINT_DATA\t ' + num_cells + '\n')
                    else:
                        outfile.write(line)
            infile.close()
            outfile.close()
            vtk_filename = out_dir + '/' + fle.split('/')[-1]
            if not os.path.exists(os.path.dirname(vtk_filename)):
                os.makedirs(os.path.dirname(vtk_filename))
            arg_string = os.environ['VTK_PATH'] + ' ' + jobname + inp_file + ' ' + jobname + vtk_filename
            subprocess.call(arg_string, shell=True)
            arg_string = 'tail -n +6 ' + jobname + temp_file + ' > ' + jobname + temp_file + '.tmp && mv ' + jobname + temp_file + '.tmp ' + jobname + temp_file
            subprocess.call(arg_string, shell=True)
            arg_string = 'cat ' + jobname + temp_file + ' >> ' + jobname + vtk_filename
            subprocess.call(arg_string, shell=True)

        print('--> Parsing PFLOTRAN output complete')

    def inp2vtk_python(self, inp_file=''):
        import pyvtk as pv
        """ Using Python VTK library, convert inp file to VTK file.  then change name of CELL_DATA to POINT_DATA.
        """
        print("--> Using Python to convert inp files to VTK files")
        if self._inp_file:
            inp_file = self._inp_file
        else:
            self._inp_file = inp_file

        if inp_file == '':
            sys.exit('ERROR: Please provide inp filename!')

        if self._vtk_file:
            vtk_file = self._vtk_file
        else:
            vtk_file = inp_file[:-4]
            self._vtk_file = vtk_file + '.vtk'

        print("--> Reading inp data")

        with open(inp_file, 'r') as f:
            line = f.readline()
            num_nodes = int(line.strip(' ').split()[0])
            num_elems = int(line.strip(' ').split()[1])

            coord = np.zeros((num_nodes, 3), 'float')
            elem_list_tri = []
            elem_list_tetra = []

            for i in range(num_nodes):
                line = f.readline()
                coord[i, 0] = float(line.strip(' ').split()[1])
                coord[i, 1] = float(line.strip(' ').split()[2])
                coord[i, 2] = float(line.strip(' ').split()[3])

            for i in range(num_elems):
                line = f.readline().strip(' ').split()
                line.pop(0)
                line.pop(0)
                elem_type = line.pop(0)
                if elem_type == 'tri':
                    elem_list_tri.append([int(i) - 1 for i in line])
                if elem_type == 'tet':
                    elem_list_tetra.append([int(i) - 1 for i in line])

        print('--> Writing inp data to vtk format')

        vtk = pv.VtkData(pv.UnstructuredGrid(coord, tetra=elem_list_tetra, triangle=elem_list_tri),
                         'Unstructured pflotran grid')
        vtk.tofile(vtk_file)

    def parse_pflotran_vtk_python(self, grid_vtk_file=''):
        """ Replace CELL_DATA with POINT_DATA in the VTK output."""
        print('--> Parsing PFLOTRAN output with Python')
        if grid_vtk_file:
            self._vtk_file = grid_vtk_file
        else:
            self.inp2vtk_python()

        grid_file = self._vtk_file

        files = glob.glob('*-[0-9][0-9][0-9].vtk')
        with open(grid_file, 'r') as f:
            grid = f.readlines()[3:]

        out_dir = 'parsed_vtk'
        for line in grid:
            if 'POINTS' in line:
                num_cells = line.strip(' ').split()[1]

        for file in files:
            with open(file, 'r') as f:
                pflotran_out = f.readlines()[4:]
            pflotran_out = [w.replace('CELL_DATA', 'POINT_DATA ') for w in pflotran_out]
            header = ['# vtk DataFile Version 2.0\n',
                      'PFLOTRAN output\n',
                      'ASCII\n']
            filename = out_dir + '/' + file
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            with open(filename, 'w') as f:
                for line in header:
                    f.write(line)
                for line in grid:
                    f.write(line)
                f.write('\n')
                f.write('\n')
                if 'vel' in file:
                    f.write('POINT_DATA\t ' + num_cells + '\n')
                for line in pflotran_out:
                    f.write(line)
        print('--> Parsing PFLOTRAN output complete')

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
                         'full_mesh_viz.inp', 'full_mesh_vol_area', 'materialid.dat','parsed_vtk', 'perm.dat',
                         'pboundary_', 'convert_uge_params.txt'] #
        move_files(gen_list, 'DFN_generator')
        move_files(lagrit_list, 'LaGriT')
        move_files(pflotran_list, 'PFLOTRAN')


if __name__ == '__main__':

    jobname, forward_input_file, ncpu = commandline_parser(sys.argv[1:])

    dfn = DFNWORKS(jobname=jobname, ncpu=int(ncpu))
    dfn._dfnGen_file = forward_input_file + '/gen_user_ellipses.dat'
    dfn._local_dfnGen_file = 'gen_user_ellipses.dat'
    dfn._aper_file = 'aperture.dat'
    dfn._perm_file = 'perm.dat'

    dfn.dfn_gen()
    dfn.lagrit2pflotran()
    flow_inputs = [f for f in os.listdir(forward_input_file) if re.match(r'dfn_explicit_.*.in', f)]
    tic_flow = time()
    for f in flow_inputs:
        tic = time()
        print('{0}\nProcessing PFLOTRAN input file: {1}\n{0}'.format('='*60, f))
        dfn._dfnFlow_file = forward_input_file + '/' + f
        dfn._local_dfnFlow_file = f
        bc = f.split('.')[0].split('_')
        save_path = '_'.join(bc[2:])

        dfn.pflotran()
        dfn.parse_pflotran_vtk_python()
        dfn.pflotran_cleanup()
        toc = time()
        with open(dfn._jobname + '/run_time_log.txt', 'a') as run_time_log:
            run_time_log.write('PFLOTRAN On Boundary {0} - {1} finished within {2:.4f} sec\n'.format(bc[2], bc[3], toc-tic))
    toc_flow = time()
    with open(dfn._jobname + '/run_time_log.txt', 'a') as run_time_log:
        run_time_log.write('Process: dfnFlow: ' + str(toc_flow - tic_flow) + ' seconds\n')
    # dfn.dfn_flow()
    dfn.clean_up_files()

