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
solver = 'fluent' # can be star, ...!TODO add CFX, OpenFoam, ...
vtuFile = r''  # path of the 3D mesh in .vtu format. Only required for Simvascular


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

if solver == 'star':
    # write .csv for star-ccm+
    with open(osp.join(profilesDir, saveName + '.csv'), 'w') as fn:
        row = 'X,Y,Z'
        for j in range(len(tcfd)):
            row += ',u(m/s)[t={}s],v(m/s)[t={}s],w(m/s)[t={}s]'.format(tcfd[j], tcfd[j], tcfd[j])
        row += '\n'
        fn.write(row)
        for i in tqdm(range(len(pos))):
            row = '{},{},{}'.format(pos[i, 0], pos[i, 1], pos[i, 2])
            for j in range(len(tcfd)):
                row += ',{},{},{}'.format(velcfd[j, i, 0], velcfd[j, i, 1], velcfd[j, i, 2])
            row += '\n'
            fn.write(row)

if solver == 'fluent':
    # write .prof for ansys fluent
    xx, yy, zz = pos[:, 0].tolist(), pos[:, 1].tolist(), pos[:, 2].tolist()
    fu = np.swapaxes(velcfd[:, :, 0], 0, 1)
    fv = np.swapaxes(velcfd[:, :, 1], 0, 1)
    fw = np.swapaxes(velcfd[:, :, 2], 0, 1)
    for i in tqdm(range(len(tcfd))):
        with open(osp.join(profilesDir, saveName + '_{:05d}.prof'.format(i)), 'w') as fn:
            fn.write('((velocity point {})\n'.format(npts))
            fn.write('(x\n')
            for xi in xx:
                fn.write(str(xi) + '\n')
            fn.write(')\n')
            fn.write('(y\n')
            for yi in yy:
                fn.write(str(yi) + '\n')
            fn.write(')\n')
            fn.write('(z\n')
            for zi in zz:
                fn.write(str(zi) + '\n')
            fn.write(')\n')
            fn.write('(u\n')
            for ui in fu[:, i]:
                fn.write(str(ui) + '\n')
            fn.write(')\n')
            fn.write('(v\n')
            for vi in fv[:, i]:
                fn.write(str(vi) + '\n')
            fn.write(')\n')
            fn.write('(w\n')
            for wi in fw[:, i]:
                fn.write(str(wi) + '\n')
            fn.write(')\n')
            fn.write(')')


if solver == 'simvascular':
    # write bct.dat for simvascular
    if saveName != 'bct': saveName = 'bct'
    mesh3d = pv.read(vtuFile)
    idx = [mesh3d.find_closest_point(pos[i]) for i in range(len(pos))]
    xx, yy, zz = pos[:, 0].tolist(), pos[:, 1].tolist(), pos[:, 2].tolist()
    fu = np.swapaxes(velcfd[:, :, 0], 0, 1)
    fv = np.swapaxes(velcfd[:, :, 1], 0, 1)
    fw = np.swapaxes(velcfd[:, :, 2], 0, 1)
    np, nl = len(pos), len(tcfd)
    with open(osp.join(profilesDir, saveName + '.dat'), 'w') as fn:
        row = '{} {}\n'.format(np, nl)
        fn.write(row)
        for i in tqdm(range(np)):
            row = '{} {} {} {} {}\n'.format(xx[i], yy[i], zz[i], nl, idx[i])
            fn.write(row)
            for j in range(nl):
                row = '{} {} {} {}\n'.format(fu[i, j], fv[i, j], fw[i, j], tcfd[j])
                fn.write(row)


