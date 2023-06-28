import os
import os.path as osp
import re
import sys
import utils as ut

import numpy as np
import pyvista as pv
from tqdm import tqdm


#-----------------------------------------------------------------------------------------------------------------------
## Options
dataDir = r''
saveDir = r''
subject_id = ''
venc = [0] * 3
flipX, flipY, flipZ = False, False, False
minusU, minusV, minusW = False, False, False
write_pcmra = True
phaseCoeff = 1.0
magCoeff = 10.0


#-----------------------------------------------------------------------------------------------------------------------
## Read dicom files and create numpy arrays
data, meta = ut.read_acquisition(dataDir)
arrayData = ut.seriesData_to_arrayData(data, meta)

#-----------------------------------------------------------------------------------------------------------------------
## Automatic detection of magnitude series
meansArr = [np.mean(x) for x in arrayData]
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

magId = np.argmax(allmean)
magTemp = arrayData[magId]
arrayData.pop(magId)
velTemp = np.zeros((meta['num_rows'], meta['num_cols'], meta['num_slices'], meta['num_frames'], 3))
for i in range(3):
    velTemp[:, :, :, :, i] = arrayData[i][:]


#-----------------------------------------------------------------------------------------------------------------------
## Velocity adjustment
print('Adjusting units.')
if re.search('GE', meta['vendor'], re.IGNORECASE):
    velTemp *= 0.001  # m/s
elif re.search('siemens', meta['vendor'], re.IGNORECASE) or re.search('philips', meta['vendor'], re.IGNORECASE):
    levels = 2**meta['HighBit']-1
    for d in range(3):
        velTemp[:, :, :, :, d] = (velTemp[:, :, :, :, d] - levels) * venc[d] / levels
    velTemp *= 0.01  # m/s
else:
    print('Manufacturer not found. Exiting.')
    sys.exit()


if meta['position'] == 'FFS':
    velTemp[:, :, :, :, 2] *= -1

if meta['position'] == 'HFS':
    velTemp[:, :, :, :, 0] *= -1
    velTemp[:, :, :, :, 1] *= -1

velTemp = np.flip(velTemp, 0)
magTemp = np.flip(magTemp, 0)
velTemp = np.flip(velTemp, 2)
magTemp = np.flip(magTemp, 2)

magTemp = np.swapaxes(magTemp, 0, 2)
velTemp = np.swapaxes(velTemp, 0, 2)

if flipX:
    velTemp = np.flip(velTemp, 0)
    magTemp = np.flip(magTemp, 0)
if flipY:
    velTemp = np.flip(velTemp, 1)
    magTemp = np.flip(magTemp, 1)
if flipZ:
    velTemp = np.flip(velTemp, 2)
    magTemp = np.flip(magTemp, 2)

if minusU:
    velTemp[:, :, :, :, 0] *= -1
if minusV:
    velTemp[:, :, :, :, 1] *= -1
if minusW:
    velTemp[:, :, :, :, 2] *= -1


#-----------------------------------------------------------------------------------------------------------------------
## Create vtk grids
os.makedirs(osp.join(saveDir, 'flow'), exist_ok=True)
print('Creating grids and writing to file.')
for f in tqdm(range(meta['num_frames']), desc='Processing and saving frames'):
    mag = magTemp[:, :, :, f]
    u   = velTemp[:, :, :, f, 0]
    v   = velTemp[:, :, :, f, 1]
    w   = velTemp[:, :, :, f, 2]

    grid = pv.UniformGrid()
    grid.dimensions = np.array(mag.shape)
    grid.origin = meta['origin']
    grid.spacing = meta['spacing'][::-1]
    grid['MagnitudeSequence'] = mag.flatten(order='F')
    grid['Velocity'] = np.transpose(np.vstack((u.flatten(order='F'),
                                               v.flatten(order='F'),
                                               w.flatten(order='F'))))

    grid.save(osp.join(saveDir, 'flow', subject_id + '_{:02d}'.format(f) + '.vtk'), binary=True)


if write_pcmra:
    print('Writing PCMRA image.')
    pcmra = np.zeros((magTemp.shape[0], magTemp.shape[1], magTemp.shape[2]))
    for i in range(meta['num_frames']):
        mag = magTemp[:, :, :, i]
        phase1 = velTemp[:, :, :, i, 0]
        phase2 = velTemp[:, :, :, i, 1]
        phase3 = velTemp[:, :, :, i, 2]

        pcmra += (magCoeff * mag) ** 2 * (
                    (phaseCoeff * phase1) ** 2 + (phaseCoeff * phase2) ** 2 + (phaseCoeff * phase3) ** 2)

    pcmra /= meta['num_frames']
    pcmra = pcmra ** 0.5

    # save pcmra image
    pcmravtk = pv.wrap(pcmra)
    pcmravtk.SetSpacing(meta['spacing'][::-1])
    pcmravtk.save(os.path.join(saveDir, subject_id + '_pcmra.vtk'))


fout = osp.join(saveDir, 'meta.txt')
fo = open(fout, "w")
for k, v in meta.items():
    fo.write(str(k) + ':  '+ str(v) + '\n')
fo.close()


print('\nFiles written to: ', osp.abspath(saveDir))


