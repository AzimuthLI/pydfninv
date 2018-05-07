import sys, os, subprocess, glob
import time, vtk
import numpy as np
import pandas as pd

def read_observation(self, obs_points, variable_name):

    '''

    :param self: DFN_INVERSE object
    :param obs_points: list of observation point coordinate, e.g. [(x1, y1, z1), ...]
    :param variable_name: the scalar variable to be observed
    :return: the observed data stored in Pandas Dataframe
    '''

    mesh_file_path = self._forward_model + '/full_mesh.vtk'
    flow_files_path = self._forward_model + '/PFLOTRAN/parsed_vtk/'

    obs_ids = get_observation_ids(mesh_file_path, obs_points)
    obs_scalars = get_observation_scalars(flow_files_path, obs_ids, variable_name)

    # Store varialbe value in Dataframe and plot
    df_scalar = pd.DataFrame.from_dict(data=obs_scalars, orient='index', dtype=np.float32)
    i = 1
    columns_name = []
    while i <= len(obs_points):
        columns_name.append('obs_'+str(i))
        i += 1

    df_scalar.columns = columns_name
    df_scalar.to_csv(self._forward_model+"obs_readings.csv")

    return df_scalar

# Get the id of observation points in mesh file
def get_observation_ids(mesh_file, obs_points):

    # Read mesh file (*.vtk)
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(mesh_file)
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
            # point_coord[key_value].append(output.GetPoint(i))
            i += 1
        id = np.argmin(dist)
        obs_ids.append(id)
    return obs_ids

def get_observation_scalars(flow_files, obs_ids, var_name):
    path, dirs, files = os.walk(flow_files).next()
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
    writer.SetFileName(self._forward_model+'/obs_points.vtk')
    writer.Update()





