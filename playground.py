import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import vtk
import vtk
from vtk.util.misc import vtkGetDataRoot

def test(**kwargs):
    a, b, c = kwargs.get('arg', [10, 20, 30])
    print(a, b, c)


if __name__ == '__main__':

    test(arg=[3,4,5])


