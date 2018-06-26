import matplotlib as mpl
import vtk, re, pickle
import numpy as np
from IPython.display import set_matplotlib_formats, Image


def nbplotstyle(style='seaborn-pastel', figsize=[9, 6]):
    set_matplotlib_formats('retina')
    mpl.style.use(style)
    params = {'backend': 'ps',
              'axes.labelsize': 12,  # fontsize for x and y labels (was 10)
              'axes.titlesize': 14,
              'axes.labelweight': 'bold',
              'font.size': 12,  # was 10
              'legend.fontsize': 12,  # was 10
              'xtick.labelsize': 12,
              'ytick.labelsize': 12,
              'figure.figsize': figsize,
              'font.family': 'Sans-serif'
              }
    mpl.rcParams.update(params)


def parse_jobreport(job_report):
    report = open(job_report, 'r').readlines()
    separator = '== RICHARDS FLOW {}\n'.format('='*63)
    idx_sep = [i for i, j in enumerate(report) if j == separator]
    idx_sep.append(idx_sep[-1]+12)

    time_str = []
    writevtk = []
    pattern_time = re.compile(r' Step(.*)Time=(.*)Dt=.*')
    pattern_writevtk = re.compile(r' --> write rate output file: dfn_explicit-darcyvel-(.*)')
    for i in range(len(idx_sep)-1):
        vtk_flag = []
        for line in report[idx_sep[i]:idx_sep[i+1]]:
            matchObj_time = pattern_time.match(line)
            matchObj_writevtk = pattern_writevtk.match(line)
            if matchObj_time:
                time_str.append(matchObj_time.group(2))
            if matchObj_writevtk:
                vtk_flag.append(True)
            else:
                vtk_flag.append(False)
        if any(vtk_flag):
            writevtk.append(True)
        else:
            writevtk.append(False)
    time = np.asarray([float(e) for e in time_str], dtype=np.float64)
    return time[writevtk]


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


def load_chain(filepath):
    rms_chain = []
    id_chain = []
    shape_vars = []
    f = open(filepath, 'rb')
    while True:
        try:
            state = pickle.load(f)
            rms_chain.append(state['rms'])
            shape_vars.append(state['dfn'])
            id_chain.append(state['dfn_id'])
        except EOFError:
            break
    f.close()

    return rms_chain, shape_vars, id_chain

