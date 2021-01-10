#!/home/simone/phd/coding/flow4D/venv/bin python3

## imports

from itertools import groupby
import re

from utils import *
from readers import flowReader

## preferences

datadir         = ''
units           = 'm/s'
saveFormat      = '.vtk'
outdir          = './processed'
saveName        = ''
prefix          = 'flow4D_'
venc            = [200, 200, 200]
reading_method  = 'series'  # choose between 'folders' and 'series'. Use 'folders' if data is already organized.


## read files from folder and dicom tags

reader = flowReader()
reader.read(datadir, method=reading_method)
ds = reader.ds
series0 = reader.series0
series1 = reader.series1
series2 = reader.series2
series3 = reader.series3


#get number of frames
K = []
for k,v in groupby(series0, key=lambda x:x['SliceLocation']):
    K.append(k)

vendor      = ds.Manufacturer
slices      = len(set(K))
frames      = len(series0) // slices
rows        = ds.Rows
columns     = ds.Columns
#origin      = ds.ImagePositionPatient
origin      = [0.0, 0.0, 0.0]
orientation = ds.ImageOrientationPatient
position    = ds.PatientPosition
period      = float(ds.NominalInterval) / 1000
spacing     = [float(ds.PixelSpacing[1]), float(ds.PixelSpacing[0]), get_dz(ds)]
if units == 'm/s':
    spacing = [s / 1000 for s in spacing]

series0 = sorted(series0, key=lambda k: k['FileName'])
series1 = sorted(series1, key=lambda k: k['FileName'])
series2 = sorted(series2, key=lambda k: k['FileName'])
series3 = sorted(series3, key=lambda k: k['FileName'])

## organize data

print('Reorganizing data.')

arr0 = organizeSeries(series0, rows, columns, slices, frames)
arr1 = organizeSeries(series1, rows, columns, slices, frames)
arr2 = organizeSeries(series2, rows, columns, slices, frames)
arr3 = organizeSeries(series3, rows, columns, slices, frames)
allArr = [arr0, arr1, arr2, arr3]

# detect magnitude series
meansArr = [np.mean(x) for x in allArr]
dist0 = []
dist1 = []
dist2 = []
dist3 = []
for i in [1, 2, 3]:
        dist0.append(np.abs(meansArr[0] - meansArr[i]))
for i in [0, 2, 3]:
        dist1.append(np.abs(meansArr[1] - meansArr[i]))
for i in [0, 1, 3]:
        dist2.append(np.abs(meansArr[2] - meansArr[i]))
for i in [0, 1, 2]:
        dist3.append(np.abs(meansArr[3] - meansArr[i]))

allmean = [np.mean(d) for d in [dist0, dist1, dist2, dist3]]

'''
clusters, centroids = kmeans1d.cluster([np.mean(x) for x in allArr], 2)
vals, idx_start, count = np.unique(clusters, return_counts=True, return_index=True)
magId = clusters[int(vals[count == 1])]
'''

magId = np.argmax(allmean)
magTemp = allArr[magId]
allArr.pop(magId)
velTemp = np.zeros((rows, columns, slices, frames, 3))
for i in range(3):
    velTemp[:, :, :, :, i] = allArr[i][:]


# adjust velocity
print('Adjusting units.')
if re.search('GE', vendor, re.IGNORECASE):
    if units == 'm/s':
        velTemp = velTemp * 0.001  # m/s
elif re.search('siemens', vendor, re.IGNORECASE) or re.search('philips', vendor, re.IGNORECASE):
    levels = 2**ds.HighBit-1
    for d in range(3):
        velTemp[:, :, :, :, d] = (velTemp[:, :, :, :, d] - levels) * venc[d] / levels
    if units == 'm/s':
        velTemp = velTemp * 0.01  # m/s
else:
    print('Manufacturer not found. Exiting.')
    sys.exit()

# adjust directions -- CAREFUL, not throroughly checked!
print('Adjusting directions.')

if reader.FOLDER_BASED_FLAG:
    velTemp_new = velTemp
    if reader.minusX:
        velTemp_new[:, :, :, :, 0] = -velTemp[:, :, :, :, 0]
    if reader.minusY:
        velTemp_new[:, :, :, :, 1] = -velTemp[:, :, :, :, 1]
    if reader.minusZ:
        velTemp_new[:, :, :, :, 2] = -velTemp[:, :, :, :, 2]
    if position == 'FFS':
        velTemp_new = np.flip(velTemp_new, 2)
        magTemp = np.flip(magTemp, 2)
    velTemp = velTemp_new
else:
    velTemp_new = np.zeros_like(velTemp)
    if position == 'HFS':
        velTemp_new[:, :, :, :, 0] = -velTemp[:, :, :, :, 2]
        velTemp_new[:, :, :, :, 1] = velTemp[:, :, :, :, 1]
        velTemp_new[:, :, :, :, 2] = velTemp[:, :, :, :, 0]
    elif position == 'FFS':
        velTemp_new[:, :, :, :, 0] = velTemp[:, :, :, :, 2]
        velTemp_new[:, :, :, :, 1] = velTemp[:, :, :, :, 1]
        velTemp_new[:, :, :, :, 2] = velTemp[:, :, :, :, 0]
        velTemp_new = np.flip(velTemp_new, 2)
        magTemp = np.flip(magTemp, 2)
    else:
        print('Patient position not found. Exiting.')
        sys.exit()
    velTemp = velTemp_new

## create grids

print('Creating grids.')

magData = []
velData = []
allData = []
for f in tqdm(range(frames), desc='Processing frame'):
    #magData.append(create_magnitude(magTemp[:,:,:,f], spacing, origin))
    #velData.append(create_velocity(velTemp[:,:,:,f,0], velTemp[:,:,:,f,1], velTemp[:,:,:,f,2], spacing, origin))
    mag = magTemp[:, :, :, f]
    u   = velTemp[:, :, :, f, 0]
    v   = velTemp[:, :, :, f, 1]
    w   = velTemp[:, :, :, f, 2]
    allData.append(create_all_vars_points(mag, u, v, w, spacing, origin))
    #allData.append(create_all_vars(mag, u, v, w, spacing, origin))

## save processed data

print('Saving data to file.')

saveDir = join(outdir, saveName)
sureDir(saveDir)
write4DData(allData, saveDir, saveFormat, prefix)


