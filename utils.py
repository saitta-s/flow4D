import os
from os.path import join
import vtk
import numpy as np
import pyvista as pv
import sys
from tqdm import tqdm


def organizeSeries(series, rows, columns, slices, frames):
    newArr = np.zeros((rows, columns, slices, frames))
    series = sorted(series, key=lambda k: k['SliceLocation'])
    ids = np.arange(0, slices*frames-frames, frames)
    for i in range(len(ids)):
        for j in range(frames):
            newArr[:, :, i, j] = series[ids[i]+j]['pixel_array']
    return newArr

def get_dz(ds):
    try:
        dz = float(ds.SpacingBetweenSlices)
    except:
        dz = float(ds.SliceThickness)
    return dz

def create_magnitude(img_arr, spacing, origin):
    mesh = pv.UniformGrid()
    mesh.dimensions = np.array(img_arr.shape) + 1
    mesh.origin = origin  # The bottom left corner of the data set
    mesh.spacing = spacing # These are the cell sizes along each axis
    mesh.cell_arrays["magnitude"] = img_arr.flatten(order="F")
    return mesh

def create_velocity(u, v, w, spacing, origin):
    vel = np.sqrt(np.square(u) + np.square(v) + np.square(w))
    mesh = pv.UniformGrid()
    mesh.dimensions = np.array(u.shape)
    mesh.origin = origin  # The bottom left corner of the data set
    mesh.spacing = spacing  # These are the cell sizes along each axis
    mesh.point_arrays['velocity'] = np.transpose(np.vstack((u.flatten(order='F'),
                                                           v.flatten(order='F'),
                                                           w.flatten(order='F'))))
    mesh.point_arrays["VelocityMagnitude"] = vel.flatten(order="F")
    return mesh

def create_all_vars(mag, u, v, w, spacing, origin):
    mesh = pv.UniformGrid()
    mesh.dimensions = np.array(mag.shape) + 1
    mesh.origin = origin
    mesh.spacing = spacing
    mesh.cell_arrays['MagnitudeSequence'] = mag.flatten(order='F')
    mesh.cell_arrays['Velocity'] = np.transpose(np.vstack((u.flatten(order='F'),
                                                            v.flatten(order='F'),
                                                            w.flatten(order='F'))))
    return mesh


def create_all_vars_points(mag, u, v, w, spacing, origin):
    mesh = pv.UniformGrid()
    mesh.dimensions = np.array(mag.shape)
    mesh.origin = origin
    mesh.spacing = spacing
    mesh.point_arrays['MagnitudeSequence'] = mag.flatten(order='F')
    mesh.point_arrays['Velocity'] = np.transpose(np.vstack((u.flatten(order='F'),
                                                            v.flatten(order='F'),
                                                            w.flatten(order='F'))))
    return mesh


def sureDir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass

def write4DData(data, path, saveFormat, prefix=''):
    nOfTimeSteps = len(data)
    if 'vtk' in saveFormat:
        for fr in tqdm(range(nOfTimeSteps), desc='Saving frame'):
            writer = vtk.vtkXMLStructuredGridWriter()
            writer.SetInputData(data[fr].cast_to_structured_grid())
            writer.SetFileName(join(path, prefix + '{:02d}'.format(fr) + '.vtk'))
            writer.Update()
    elif 'vtu' in saveFormat:
        for fr in tqdm(range(nOfTimeSteps), desc='Saving frame'):
            writer = vtk.vtkXMLUnstructuredGridWriter()
            writer.SetInputData(data[fr].cast_to_unstructured_grid())
            writer.SetFileName(join(path, prefix + '{:02d}'.format(fr) + '.vtu'))
            writer.Update()
    else:
        print("Error: only .vtk and .vtu formats are supported.")
        sys.exit()



'''
import scipy.io
newdir = r'C:\DATA\phd_laptop\tesisti_magistrali\valentina\dati'

pts4df = allData[0].points
mdic = {"pts4df": pts4df}

scipy.io.savemat(os.path.join(newdir, 'pts4df.mat'), mdic)

for f in tqdm(range(frames)):
    mdic = {"vel4df": allData[f].point_arrays['Velocity']}
    sn = os.path.join(newdir, 'vel4df_%d.mat' % f)
    scipy.io.savemat(sn, mdic)
    
'''

