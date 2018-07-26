from __future__ import print_function

import vtk
from time import sleep
import os

class vtkTimerCallback():

    def __init__(self, mesh_list, txtActor):
        self.timer_count = 0
        self.mesh_file_list = mesh_list
        self.txtActor = txtActor

    def execute(self, obj, event):
        print(self.timer_count)

        txt_content = "Model ID: {}".format(self.timer_count)
        self.txtActor.SetInput(txt_content)

        reader = vtk.vtkUnstructuredGridReader()
        mesh_file = self.mesh_file_list[self.timer_count]
        reader.SetFileName(mesh_file)
        reader.Update()
        output = reader.GetOutput()
        self.mapper.SetInputData(output)

        # self.actor.SetPosition(self.timer_count, self.timer_count, 0)
        iren = obj
        iren.GetRenderWindow().Render()

        if self.timer_count >= len(self.mesh_file_list)-1:
            self.timer_count = -1
        self.timer_count += 1
        # sleep(1)


def animation(mesh_list, obs_points):

    colors = vtk.vtkNamedColors()
    # Create a mapper and actor
    mapper = vtk.vtkDataSetMapper()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    points = vtk.vtkPoints()
    for pt in obs_points:
        points.InsertNextPoint(pt)

    # Glyph the points
    sphere = vtk.vtkSphereSource()
    sphere.SetPhiResolution(21)
    sphere.SetThetaResolution(21)
    sphere.SetRadius(.01)

    # Create a polydata to store everything in
    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points)

    pointMapper = vtk.vtkGlyph3DMapper()
    pointMapper.SetInputData(polyData)
    pointMapper.SetSourceConnection(sphere.GetOutputPort())

    pointActor = vtk.vtkActor()
    pointActor.SetMapper(pointMapper)
    pointActor.GetProperty().SetColor(colors.GetColor3d("Tomato"))

    # Creat StringArray to store the observation point label
    lblmapper = vtk.vtkLabeledDataMapper()
    lblmapper.SetInputData(polyData)
    lbl_text_property = vtk.vtkTextProperty()
    lbl_text_property.SetFontSize(20)
    lbl_text_property.SetColor(colors.GetColor3d("Tomato"))
    lblmapper.SetLabelTextProperty(lbl_text_property)
    lblActor = vtk.vtkActor2D()
    lblActor.SetMapper(lblmapper)

    # Setup a renderer, render window, and interactor
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetSize(2000, 2000)

    light_touse = vtk.vtkLight()
    light_touse.SetDirectionAngle(-20, 45)

    camera_touse = vtk.vtkCamera()
    camera_touse.Elevation(-80)
    camera_touse.Roll(-30)
    camera_touse.Azimuth(30)
    camera_touse.Dolly(0.25)
    renderer.SetActiveCamera(camera_touse)
    renderer.AddLight(light_touse)

    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    transform = vtk.vtkTransform()
    transform.Translate(-0.8, -1.0, -0.2)
    axes = vtk.vtkAxesActor()
    #  The axes are positioned with a user transform
    axes.SetUserTransform(transform)
    axes.SetTotalLength(0.25, 0.25, 0.25)
    axes.SetShaftTypeToLine()
    axes.GetXAxisCaptionActor2D().GetTextActor().SetTextScaleMode(10)
    axes.GetYAxisCaptionActor2D().GetTextActor().SetTextScaleMode(10)
    axes.GetZAxisCaptionActor2D().GetTextActor().SetTextScaleMode(10)
    axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(colors.GetColor3d("Red"))
    axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetColor(colors.GetColor3d("Green"))
    axes.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetColor(colors.GetColor3d("Blue"))

    txtActor = vtk.vtkTextActor()
    txtprop = txtActor.GetTextProperty()
    txtprop.SetFontFamilyToArial()
    txtprop.BoldOn()
    txtprop.SetFontSize(36)
    txtprop.SetShadowOffset(4, 4)
    txtprop.SetColor(colors.GetColor3d("r"))
    txtActor.SetDisplayPosition(1000, 1200)

    # Add the actor to the scene
    renderer.AddActor(actor)
    renderer.AddActor(pointActor)
    renderer.AddActor(lblActor)
    renderer.AddActor(axes)
    renderer.AddActor(txtActor)
    renderer.SetBackground(1, 1, 1)  # Background color white

    # Render and interact
    renderWindow.Render()

    # Initialize must be called prior to creating timer events.
    renderWindowInteractor.Initialize()

    # Sign up to receive TimerEvent
    cb = vtkTimerCallback(mesh_list, txtActor)
    cb.mapper = mapper
    renderWindowInteractor.AddObserver('TimerEvent', cb.execute)
    renderWindowInteractor.CreateRepeatingTimer(500)

    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renderWindow)
    windowToImageFilter.SetInputBufferTypeToRGBA()
    windowToImageFilter.ReadFrontBufferOff()
    windowToImageFilter.Update()

    # writer = vtk.vtkOggTheoraWriter()
    # writer.SetInputConnection(windowToImageFilter.GetOutputPort())
    # writer.SetFileName("/Volumes/SD_Card/Thesis_project/test.ogv")
    # writer.Start()
    # renderWindow.Start()

    # start the interaction and timer
    renderWindowInteractor.Start()

    # windowToImageFilter.Modified()
    # writer.Write()
    # writer.End()


if __name__ == '__main__':

    # model_path = '/Volumes/SD_Card/Thesis_project/model_4/accept_models'
    # model_path =  '/Users/shiyili/euler_remote/home/model_1x1x1_cxyz_500/inverse/accept_models'
    model_path = '/Users/shiyili/euler_remote/scratch/dfn2_size2_n/inverse/accept_models'
    mesh_list = []
    for root, dirs, files in os.walk(model_path):
        if 'full_mesh.vtk' in files:
            mesh_list.append(os.path.join(root, 'full_mesh.vtk'))

    # obs_pt = [(0.4, 0.4, 0.2),
    #           (0.4, 0.4, -0.2),
    #           (0.4, -0.4, 0.2),
    #           (0.4, -0.4, -0.2),
    #           (-0.15, -0.08, 0.2),
    #           (-0.15, -0.08, 0)]

    # obs_pt = [(-0.5, 2, 0.5), (-1.1, 4, 2.1), (-3, -2, 1), (2, -1, 4),
    #           (-4, 2, -2), (2, -2, 2), (3, 2, -3)]

    obs_pt = [(-0.05, 0.2, 0.05), (-0.11, 0.4, 0.21), (-0.3, -0.2, 0.1), (0.2, -0.1, 0.4),
              (-0.4, 0.2, -0.2), (0.2, -0.2, 0.2), (0.3, 0.2, -0.3)]
    animation(mesh_list, obs_pt)
