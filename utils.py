import sys
import os
from os.path import join
import numpy as np
from itertools import groupby
from tqdm import tqdm
import pydicom
from collections import Counter
import re
from scipy.interpolate import RBFInterpolator, NearestNDInterpolator
from scipy.spatial import distance
import pyvista as pv
import vtk
from vmtk import vmtkscripts


def get_dz(ds):
    try:
        dz = float(ds.SpacingBetweenSlices)
    except:
        dz = float(ds.SliceThickness)
    return dz

def get_venc(data):
    venc = [0] * 3
    # Check venc from the sequence name (e.g. fl3d1_v150fh)
    j = 0

    if hasattr(data['series0'][0]['info'], 'SequenceName'):
        pattern = re.compile(".*?_v(\\d+)(\\w+)")
        for i in range(4):
            ser = data['series' + str(i)]
            found = pattern.search(ser[0]['info'].SequenceName)
            if found:
                venc[j] = int(found.group(1))
                j += 1

    elif hasattr(data['series0'][0]['info'], 'SeriesDescription'):
        pattern = re.compile(".*?VENC (\\d+).*?")
        for i in range(3):
            ser = data['series' + str(i)]
            found = pattern.search(ser[0]['info'].SeriesDescription)
            if found:
                venc[j] = int(found.group(1))
                j += 1
    print('Detected venc:', venc)
    return venc


def read_acquisition(dataDir):
    series0 = []
    series1 = []
    series2 = []
    series3 = []
    series = []
    sNum = []
    for root, dirs, files in os.walk(dataDir):
        for file in tqdm(files, desc='Reading images', disable=len(files) == 0):
            ds = pydicom.dcmread(join(root, file), force=True)
            sNum.append(ds.SeriesNumber)
            dataTemp = dict()
            dataTemp['FileName'] = file
            dataTemp['pixel_array'] = ds.pixel_array.astype('float')
            dataTemp['info'] = ds
            series.append(dataTemp)

    counter = Counter(sNum)
    sNum = np.unique(sNum)

    if len(counter) == 4:
        for i in range(len(series)):
            if int(series[i]['info'].SeriesNumber) == sNum[0]:
                series0.append(series[i])
            elif int(series[i]['info'].SeriesNumber) == sNum[1]:
                series1.append(series[i])
            elif int(series[i]['info'].SeriesNumber) == sNum[2]:
                series2.append(series[i])
            elif int(series[i]['info'].SeriesNumber) == sNum[3]:
                series3.append(series[i])
            else:
                print('Series number not found.')
                print(series[i]['info'].SeriesNumber)
                sys.exit(0)

    elif len(counter) == 2:
        num_imgs = list(counter.values())

        if num_imgs[0] > num_imgs[1]:
            assert num_imgs[0] == 3 * num_imgs[1]
            series_count = 0
            for i in range(len(series)):
                if int(series[i]['info'].SeriesNumber) == sNum[0]:
                    if series_count < num_imgs[1]:
                        series0.append(series[i])
                        series_count += 1
                    elif series_count < 2 * num_imgs[1]:
                        series1.append(series[i])
                        series_count += 1
                    elif series_count < 3 * num_imgs[1]:
                        series2.append(series[i])
                        series_count += 1
                else:
                    series3.append(series[i])

        if num_imgs[0] < num_imgs[1]:
            assert num_imgs[1] == 3 * num_imgs[0]
            series_count = 0
            for i in range(len(series)):
                if int(series[i]['info'].SeriesNumber) == sNum[1]:
                    if series_count < num_imgs[1]:
                        series0.append(series[i])
                        series_count += 1
                    elif series_count < 2 * num_imgs[1]:
                        series1.append(series[i])
                        series_count += 1
                    elif series_count < 3 * num_imgs[1]:
                        series2.append(series[i])
                        series_count += 1
                else:
                    series3.append(series[i])

    K = []
    for k, v in groupby(series0, key=lambda x: x['info'].SliceLocation):
        K.append(k)

    vendor = ds.Manufacturer
    slices = len(set(K))
    frames = len(series0) // slices
    rows = ds.Rows
    columns = ds.Columns
    # origin      = ds.ImagePositionPatient
    origin = [0.0, 0.0, 0.0]
    orientation = ds.ImageOrientationPatient
    position = ds.PatientPosition
    # period      = float(ds.NominalInterval) / 1000
    spacing = [float(ds.PixelSpacing[1]), float(ds.PixelSpacing[0]), get_dz(ds)]
    spacing = [s / 1000 for s in spacing]

    series0 = sorted(series0, key=lambda k: k['FileName'])
    series1 = sorted(series1, key=lambda k: k['FileName'])
    series2 = sorted(series2, key=lambda k: k['FileName'])
    series3 = sorted(series3, key=lambda k: k['FileName'])

    meta = {'vendor': vendor,
            'num_slices': slices,
            'num_frames': frames,
            'num_rows': rows,
            'num_cols': columns,
            'origin': origin,
            'orientation': orientation,
            'position': position,
            'spacing': spacing,
            'HighBit': ds.HighBit
    }

    series_data = {'series0': series0,
            'series1': series1,
            'series2': series2,
            'series3': series3
    }

    # venc detection
    venc = get_venc(series_data)
    if np.mean(venc) > 80:
        venc = [vv * 0.01 for vv in venc]
    meta['venc'] = venc

    return series_data, meta



def seriesData_to_arrayData(seriesData, meta):
    arrayData = []
    for s in seriesData.keys():
        series = seriesData[s]
        newArr = np.zeros((meta['num_rows'], meta['num_cols'], meta['num_slices'], meta['num_frames']))
        try:
            #IPP = []
            for j in range(1, meta['num_frames'] + 1):
                frameBlock = [elem for elem in series if int(elem['info'].TemporalPositionIdentifier) == j]
                frameBlock = sorted(frameBlock, key=lambda k: k['info'].SliceLocation)
                for i in range(meta['num_slices']):
                    newArr[:, :, i, j - 1] = frameBlock[i]['pixel_array']
                    #IPP.append(frameBlock[i]['IPP'])
            arrayData.append(newArr)
        except:
            series = sorted(series, key=lambda k: k['info'].SliceLocation)
            #series = sorted(series, key=lambda k: k['FileName'])
            ids = np.arange(0, meta['num_slices'] * meta['num_frames'] - meta['num_frames'], meta['num_frames'])
            for i in range(len(ids)):
                for j in range(meta['num_frames']):
                    newArr[:, :, i, j] = series[ids[i] + j]['pixel_array']
            arrayData.append(newArr)

    return arrayData


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix



# core function for interpolating profiles in 3D space
def interpolate_profiles(aligned_planes, fxdpts, intp_options):
    num_frames = len(aligned_planes)

    # Set boundary vectors to zero
    dr = intp_options['zero_boundary_dist']  # percentage threshold for zero boundary
    edge_pts = [aligned_planes[k].extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False).points for k in range(num_frames)]
    dist2edge = [distance.cdist(aligned_planes[k].points, edge_pts[k]).min(axis=1) for k in range(num_frames)]
    boundary_ids = [np.where(dist2edge[k] < (dr * dist2edge[k].max()))[0] for k in range(num_frames)]
    for k in range(num_frames):
        aligned_planes[k]['Velocity'][boundary_ids[k], :] = 0.0

    # Set backflow to zero
    if intp_options['zero_backflow']:
        normals = [aligned_planes[k].compute_normals()['Normals'].mean(0) * -1 for k in
                   range(num_frames)]  # Careful with the sign
        normals = [normals[k] / np.linalg.norm(normals[k]) for k in range(num_frames)]
        for k in range(num_frames):
            signs = np.dot(aligned_planes[k]['Velocity'], normals[k])
            aligned_planes[k]['Velocity'][np.where(signs < 0)] = 0.0

    # interpolate velocity profile
    vel_interp = []
    # print('fitting...')
    for k in range(num_frames):
        nnVel = NearestNDInterpolator(aligned_planes[k].points, aligned_planes[k]['Velocity'])(fxdpts)
        I = RBFInterpolator(fxdpts, nnVel,
                            kernel=intp_options['kernel'], smoothing=intp_options['smoothing'],
                            epsilon=1, degree=intp_options['degree'])

        vel_interp.append(I(fxdpts))

        # vel_interp.append(nnVel)

    # create new polydatas
    interp_planes = [pv.PolyData(fxdpts).delaunay_2d(alpha=0.1) for _ in range(num_frames)]
    for k in range(num_frames):
        interp_planes[k]['Velocity'] = vel_interp[k]

    return interp_planes



def rotation_matrix_from_axis_and_angle(u, theta):
    """:arg u is axis (3 components)
       :arg theta is angle (1 component) obtained by acos of dot prod
    """

    from math import cos, sin

    R = np.asarray([[cos(theta) + u[0] ** 2 * (1 - cos(theta)),
             u[0] * u[1] * (1 - cos(theta)) - u[2] * sin(theta),
             u[0] * u[2] * (1 - cos(theta)) + u[1] * sin(theta)],
            [u[0] * u[1] * (1 - cos(theta)) + u[2] * sin(theta),
             cos(theta) + u[1] ** 2 * (1 - cos(theta)),
             u[1] * u[2] * (1 - cos(theta)) - u[0] * sin(theta)],
            [u[0] * u[2] * (1 - cos(theta)) - u[1] * sin(theta),
             u[1] * u[2] * (1 - cos(theta)) + u[0] * sin(theta),
             cos(theta) + u[2] ** 2 * (1 - cos(theta))]])

    return R

##----------------------------------------------------------------------------------------------------------------------
# Geometric analysis functions

def clean_surface(surface, size_factor=0.1):
    surfaceCleaner = vmtkscripts.vmtkSurfaceKiteRemoval()
    surfaceCleaner.Surface = surface
    surfaceCleaner.SizeFactor = size_factor
    surfaceCleaner.Execute()
    return surfaceCleaner.Surface


def fillHoles(surface, holeSize=40):
    filler = vtk.vtkFillHolesFilter()
    filler.SetInputData(surface)
    filler.SetHoleSize(holeSize)
    filler.Update()
    return filler.GetOutput()



def extract_parent_centerline(surface, dx=0.001, smoothing_iters=50, smoothing_factor=0.5):
    cl_filter = vmtkscripts.vmtkCenterlines()
    cl_filter.Surface = surface
    #cl_filter.AppendEndPoints = 1
    cl_filter.Resampling = 1
    cl_filter.ResamplingStepLength = dx
    cl_filter.Execute()

    attr = vmtkscripts.vmtkCenterlineAttributes()
    attr.Centerlines = cl_filter.Centerlines
    attr.Execute()

    geo = vmtkscripts.vmtkCenterlineGeometry()
    geo.Centerlines = attr.Centerlines
    geo.LineSmoothing = 0
    geo.OutputSmoothingLines = 0
    geo.Execute()

    smoo = vmtkscripts.vmtkCenterlineSmoothing()
    smoo.Centerlines = geo.Centerlines
    smoo.NumberOfSmoothingIterations = smoothing_iters
    smoo.SmoothingFactor = smoothing_factor
    smoo.Execute()

    return smoo.Centerlines

