## Imports
import os
from os.path import join
import vtk
from tqdm import tqdm
import numpy as np
import pandas as pd
from vtk.util.numpy_support import vtk_to_numpy
from glob import glob
import pyvista as pv

from utils import sureDir

## preferences
processed4DFlowDataDir  = r'/home/simone/phd/coding/flow4D/processed/valentinaNY'
registered_stl          = r'/home/simone/phd/tesisti/valentina_scarponi/fixed.stl'
outputDir               = r'/home/simone/phd/tesisti/valentina_scarponi/probedData'

sureDir(outputDir)
##
# Read processed 4D flow data
allData = []
processed4DFlowFiles = sorted(glob(join(processed4DFlowDataDir, '*.vtk')))
for f in tqdm(range(len(processed4DFlowFiles)), desc='Reading processed vtk frames'):
    reader = vtk.vtkXMLStructuredGridReader()
    reader.SetFileName(processed4DFlowFiles[f])
    reader.Update()
    grid = reader.GetOutput()
    allData.append(reader.GetOutput())

# Read registered stl
reader = vtk.vtkSTLReader()
reader.SetFileName(registered_stl)
reader.Update()
geo = reader.GetOutput()

# Prepare csv file
fieldnames = ['x', 'y', 'z', 'u', 'v', 'w']
df_arr = np.empty((geo.GetNumberOfPoints(), len(fieldnames)))

for f in tqdm(range(len(allData)), desc='Writing point data for frame'):
    #vel_grid = vtk.vtkUnstructuredGrid()
    vel_grid = allData[f]
    probe = vtk.vtkProbeFilter()
    probe.SetInputData(geo)
    probe.SetSourceData(vel_grid)
    probe.Update()
    geoWithVars = probe.GetOutput()
    vtk_pts = geoWithVars.GetPoints()

    ptsArr = vtk_to_numpy(vtk_pts.GetData())
    velArr = vtk_to_numpy(geoWithVars.GetPointData().GetArray('Velocity'))

    df_arr[:, 0] = ptsArr[:, 0]
    df_arr[:, 1] = ptsArr[:, 1]
    df_arr[:, 2] = ptsArr[:, 2]
    df_arr[:, 3] = velArr[:, 0]
    df_arr[:, 4] = velArr[:, 1]
    df_arr[:, 5] = velArr[:, 2]
    df = pd.DataFrame(df_arr, columns=fieldnames)
    df.to_csv(join(outputDir, 'point_data_{:02d}.csv'.format(f)), index=False)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(join(outputDir, 'probedData_{:02d}.vtp'.format(f)))
    writer.SetInputData(geoWithVars)
    writer.Update()