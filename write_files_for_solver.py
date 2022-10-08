import sys
import os
import os.path as osp
import numpy as np
import pyvista as pv
from glob import glob
from tqdm import tqdm
from scipy.interpolate import interp1d


#-----------------------------------------------------------------------------------------------------------------------
## Options
profilesDir = r''
outputDir = r''
saveName = 'inflow_profiles'
cfd_delta_t = 0.001  # simulation time steps
cardiac_cycle_period = 1.0
time_interpolation = 'cubic' # can be linear, nearest, quadratic, ...
solver = 'star' # can be star, ...!TODO add fluent, CFX, OpenFoam and more


#-----------------------------------------------------------------------------------------------------------------------
## Prepare variables
interp_planes = [pv.read(fn) for fn in sorted(glob(osp.join(profilesDir, '*.vtp')))]
num_frames = len(interp_planes)

os.makedirs(outputDir, exist_ok=True)

tcfd = np.arange(0, cardiac_cycle_period, cfd_delta_t)
t4df = np.linspace(0, cardiac_cycle_period, num_frames)
pos = interp_planes[0].points
npts = pos.shape[0]
vel4df = np.array([interp_planes[k]['Velocity'] for k in range(len(interp_planes))])
velcfd = interp1d(t4df, vel4df, axis=0, kind=time_interpolation)(tcfd)


#-----------------------------------------------------------------------------------------------------------------------
## Write files for solver

# write csv for star-ccm+
with open(osp.join(profilesDir, saveName + '.csv'), 'w') as fn:
    riga = 'X,Y,Z'
    for j in range(len(tcfd)):
        riga += ',u(m/s)[t={}s],v(m/s)[t={}s],w(m/s)[t={}s]'.format(tcfd[j], tcfd[j], tcfd[j])
    riga += '\n'
    fn.write(riga)
    for i in tqdm(range(len(pos))):
        riga = '{},{},{}'.format(pos[i, 0], pos[i, 1], pos[i, 2])
        for j in range(len(tcfd)):
            riga += ',{},{},{}'.format(velcfd[j, i, 0], velcfd[j, i, 1], velcfd[j, i, 2])
        riga += '\n'
        fn.write(riga)


