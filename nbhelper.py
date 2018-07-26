import matplotlib as mpl
import vtk, re, pickle
import numpy as np
from IPython.display import set_matplotlib_formats, Image





def vtkmesh_show(mesh_vtk, obs_vtk, **kwargs):
    colors = vtk.vtkNamedColors()
    # Setup Mesh
    ReaderMesh = vtk.vtkUnstructuredGridReader()
    ReaderMesh.SetFileName(mesh_vtk)
    ReaderMesh.Update()  # Needed because of GetScalarRange
    Mesh = ReaderMesh.GetOutput()

    MapperMesh = vtk.vtkDataSetMapper()
    MapperMesh.SetInputData(Mesh)
    ActorMesh = vtk.vtkActor()
    ActorMesh.SetMapper(MapperMesh)
    ActorMesh.GetProperty().SetColor(colors.GetColor3d("SlateGray"))

    # Setup Observation Points
    ReaderObs = vtk.vtkPolyDataReader()
    ReaderObs.SetFileName(obs_vtk)
    ReaderObs.Update()
    ObsPoints = ReaderObs.GetOutput()

    MapperObs = vtk.vtkPolyDataMapper()
    MapperObs.SetInputData(ObsPoints)
    ActorObs = vtk.vtkActor()
    ActorObs.SetMapper(MapperObs)
    ActorObs.GetProperty().SetColor(colors.GetColor3d("Tomato"))

    # Setup Camera
    cam_ele, cam_roll, cam_azi = kwargs.get('camera_position', [-80, -30, 30])
    Cam = vtk.vtkCamera()
    Cam.Elevation(cam_ele)
    Cam.Roll(cam_roll)
    Cam.Azimuth(cam_azi)
    Cam.Dolly(0.25)

    # Setup Light
    lig_ele, lig_azi = kwargs.get('light_direction', [-20, 45])
    Light = vtk.vtkLight()
    Light.SetDirectionAngle(lig_ele, lig_azi)

    # Setup Axis
    transform = vtk.vtkTransform()
    transform.Translate(-0.8, -1.0, -0.2)
    axes = vtk.vtkAxesActor()
    axes.SetUserTransform(transform)
    axes.SetTotalLength(0.25, 0.25, 0.25)
    axes.SetShaftTypeToLine()
    axes.GetXAxisCaptionActor2D().GetTextActor().SetTextScaleMode(10)
    axes.GetYAxisCaptionActor2D().GetTextActor().SetTextScaleMode(10)
    axes.GetZAxisCaptionActor2D().GetTextActor().SetTextScaleMode(10)
    axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(colors.GetColor3d("Red"))
    axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetColor(colors.GetColor3d("Green"))
    axes.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetColor(colors.GetColor3d("Blue"))

    Renderer = vtk.vtkRenderer()
    Renderer.SetBackground(1.0, 1.0, 1.0)
    Renderer.AddActor(ActorMesh)
    Renderer.AddActor(ActorObs)
    Renderer.AddActor(axes)
    Renderer.SetActiveCamera(Cam)

    w, h = kwargs.get('fig_size', [600, 400])
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetOffScreenRendering(1)
    renderWindow.AddRenderer(Renderer)
    renderWindow.SetSize(w, h)
    renderWindow.Render()

    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renderWindow)
    windowToImageFilter.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetWriteToMemory(1)
    writer.SetInputConnection(windowToImageFilter.GetOutputPort())
    writer.Write()
    data = bytes(memoryview(writer.GetResult()))

    return Image(data)





if __name__ == '__main__':
    report_file = '/Users/shiyili/euler_remote/scratch/dfn1_size2_xyz_dr0/synthetic/job_report.txt'
    t, v = parse_jobreport(report_file)
    print(t)
    print(v)