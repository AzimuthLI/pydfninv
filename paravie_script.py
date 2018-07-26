from paraview.simple import *
import os


models_dir = input('Directory for models: ')

path, dirs, files = os.walk(models_dir).next()

for d in dirs:

    legacyvtk = OpenDataFile(models_dir+'/'+d+'/full_mesh.vtk')

Show()
Render()



