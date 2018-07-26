import os, vtk, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import normal
from deprecated.helper_2 import latexify

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

    path, dirs, files = os.walk(flow_files).__next__()
    file_count = len(files)
    print("Total time steps: %d \n" % file_count)

    file_num = 1
    observed_scalars = {}
    for vtk_file in files:

        # Read the source file.
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(path + vtk_file)
        reader.Update()  # Needed because of GetScalarRange
        output = reader.GetOutput()

        key_value = 'time ' + str(file_num)
        obs_scalar = []
        for pt in obs_ids:
            obs_scalar.append(output.GetPointData().GetScalars(var_name).GetValue(pt))
            observed_scalars.update({key_value: obs_scalar})
        file_num += 1

    return observed_scalars


def gen_3d_obs_points_plot(obs_points, save_path):

    sphere = vtk.vtkSphereSource()
    sphere.SetPhiResolution(21)
    sphere.SetThetaResolution(21)
    sphere.SetRadius(0.02)

    filters = vtk.vtkAppendPolyData()

    for pt in obs_points:
        sphere.SetCenter(pt)
        sphere.Update()

        inputs = vtk.vtkPolyData()
        inputs.ShallowCopy(sphere.GetOutput())

        filters.AddInputData(inputs)

    filters.Update()

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(filters.GetOutput())
    writer.SetFileName(save_path + '/obs_points.vtk')
    writer.Update()


if __name__ == '__main__':
    #
    syn_model_path = '/cluster/home/lishi/model_1x1x1_cx_500/synthetic/output'
    #
    input_file_path = '/cluster/home/lishi/model_1x1x1_cx_500/synthetic/inputs/'

    # Model 1 (1x1x1, cx)
    observe_points = [(0.4, 0.4, 0.2), (0.4, 0.4, -0.2), (0.4, -0.4, 0.2),
                      (0.4, -0.4, -0.2), (-0.15, -0.08, 0.2), (-0.15, -0.08, 0)]

    # Model 2 (1x1x1, cxyz)
    # observe_points = [(-0.05, 0.2, 0.05), (-0.11, 0.4, 0.21), (-0.3, -0.2, 0.1), (0.2, -0.1, 0.4),
    #                   (-0.4, 0.2, -0.2), (0.2, -0.2, 0.2), (0.3, 0.2, -0.3)]

    # Model 3 (5x5x5, cx)
    # observe_points = [(2, 2, 1), (2, 2, -1), (2, -2, 1),
    #                   (2, -2, -1), (-0.75, -0.4, 1), (-0.75, -0.4, 0)]

    # Model 4 （5x5x5, cxyz)
    # observe_points = [(-0.5, 2, 0.5), (-1.1, 4, 2.1), (-3, -2, 1), (2, -1, 4),
    #                   (-4, 2, -2), (2, -2, 2), (3, 2, -3)]

    ncpu = 4
    run_dfnworks_command = ['python3', 'dfnworks.py',
                            '-j', syn_model_path,
                            '-i', input_file_path,
                            '-n', str(ncpu)
                           ]
    p = subprocess.Popen(run_dfnworks_command)
    p.wait()

    # Read pressure data from the outputs
    mesh_file_path = syn_model_path + '/full_mesh.vtk'
    flow_files_path = syn_model_path + '/PFLOTRAN/parsed_vtk/'

    variable_name = ['Liquid_Pressure']

    observe_ids = get_observation_ids(mesh_file_path, observe_points)
    obs_scalars = get_observation_scalars(flow_files_path, observe_ids, variable_name[0])

    # Store variables value in DataFrame and plot
    df_pressure = pd.DataFrame.from_dict(data=obs_scalars, orient='index', dtype=np.float32)

    i = 1
    columns_name = []
    while i <= len(observe_points):
        columns_name.append('obs_'+str(i))
        i += 1

    df_pressure.columns = columns_name

    df_pressure += normal(0, 0.1, size=df_pressure.shape)

    df_pressure.to_csv(syn_model_path+"/obs_readings.csv")
    #
    gen_3d_obs_points_plot(observe_points, syn_model_path)
    # df_pressure = pd.read_csv(syn_model_path+"/obs_readings.csv", index_col=0)
    df_pressure = df_pressure / 1e6

    # columns_name = df_pressure.columns
    # plot observation
    latexify(fig_width=5.1)
    fig = plt.figure()

    for col in columns_name:
        plt.plot(np.arange(0, len(df_pressure)), df_pressure[col], label='Station ' + col.split('_')[-1])

    plt.xlabel('Time Step')
    plt.ylabel('Pressure(MPa)')
    plt.title('Synthethic Observation')
    plt.legend(loc=7, bbox_to_anchor=(1, 0.65))
    plt.savefig(syn_model_path + '/observation.pdf')
    plt.close(fig)
    # plt.show()
