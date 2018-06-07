import vtk
from math import sqrt
from time import time
import matplotlib as mpl
SPINE_COLOR = 'gray'

def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    mpl.style.use('ggplot')

    assert (columns in [1, 2])

    if fig_width is None:
        fig_width = 3.39 if columns == 1 else 6.9  # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
        fig_height = fig_width * golden_mean  # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" +str(fig_height) +
              "so will reduce to" + str(MAX_HEIGHT_INCHES) + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {'backend': 'ps',
              'text.latex.preamble': ['\\usepackage{gensymb}'],
              'axes.labelsize': 8,  # fontsize for x and y labels (was 10)
              'axes.titlesize': 8,
              'font.size': 8,  # was 10
              'legend.fontsize': 8,  # was 10
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'text.usetex': True,
              'figure.figsize': [fig_width, fig_height],
              'font.family': 'serif'
              }

    mpl.rcParams.update(params)


def format_axes(ax):
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

    return ax


def vtk_interactiveshow(renderer, w=800, h=800):

    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(renderer)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    renWin.SetSize(w, h)

    iren.Initialize()
    renWin.Render()
    iren.Start()

def vtk_show(renderer, w=100, h=100):
    """
    Takes vtkRenderer instance and returns an IPython Image with the rendering.
    """
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetOffScreenRendering(1)
    renderWindow.AddRenderer(renderer)
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

    from IPython.display import Image
    return Image(data)

def vtk_meshrender(mesh_file, obs_points, **kwargs):

    # plot fracture network with observation points
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(mesh_file)
    reader.Update()  # Needed because of GetScalarRange
    output = reader.GetOutput()

    # Create the mapper that corresponds the objects of the vtk.vtk file
    # into graphics elements
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputData(output)

    # Create the Actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    # actor.GetProperty().EdgeVisibilityOn()

    points = vtk.vtkPoints()
    for pt in obs_points:
        points.InsertNextPoint(pt)

    colors = vtk.vtkNamedColors()
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

    # Creat StringArray to stroe the observation point label
    lblmapper = vtk.vtkLabeledDataMapper()
    lblmapper.SetInputData(polyData)
    lbl_text_property = vtk.vtkTextProperty()
    lbl_text_property.SetFontSize(20)
    lbl_text_property.SetColor(colors.GetColor3d("Tomato"))
    lblmapper.SetLabelTextProperty(lbl_text_property)
    lblActor = vtk.vtkActor2D()
    lblActor.SetMapper(lblmapper)

    # light = vtk.vtkLight()
    # light.SetFocalPoint(0.21406, 1.5, 0)
    # light.SetPosition(8.3761, 4.94858, 4.12505)

    # Create the Renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.AddActor(pointActor)
    renderer.AddActor(lblActor)

    light_touse = vtk.vtkLight()
    light_touse.SetDirectionAngle(-20, 45)

    ele, rol, azi = kwargs.get('camare_ang', [-80, -30, 30])
    camera_touse = vtk.vtkCamera()
    camera_touse.Elevation(ele)
    camera_touse.Roll(rol)
    camera_touse.Azimuth(azi)
    camera_touse.Dolly(0.25)
    renderer.SetActiveCamera(camera_touse)
    renderer.AddLight(light_touse)

    renderer.SetBackground(1, 1, 1)  # Set background to white

    return renderer

def update_progress(i, total, tic):
    progress = i / total
    toc = time()
    dt = toc - tic

    barLength = 20  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength * progress))
    text = "\rPercent: [{0}] {1:.2f}% {2}, {3} / {4} finished. Average time: {5:.5f}" \
        .format("#" * block + "-" * (barLength - block), progress * 100, status, i, total, dt / i)
    sys.stdout.write(text)
    sys.stdout.flush()