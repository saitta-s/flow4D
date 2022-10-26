

import sys
import os
import os.path as osp
import numpy as np
from glob import glob
import random
from tqdm import tqdm
import pandas as pd
import pyvista as pv
from sklearn.decomposition import PCA
from collections import deque

import utils as ut



#-----------------------------------------------------------------------------------------------------------------------
## Options
dataDir = r'C:\DATA\phd_laptop\dev\ssm_prj\SSM_output\clinical_cohort'
outDir = r'C:\DATA\phd_laptop\dev\ssm_prj\SSM_output2'

intp_options = {
    'zero_boundary_dist': 0.05,
    'zero_backflow': False,
    'kernel': 'linear',
    'smoothing': 1.5,
    'epsilon': 1,
    'degree': 0}
time_intp_options = {
    'T4df': 1,
    'Tfxd': 1,
    'num_frames_fxd': 20}


#-----------------------------------------------------------------------------------------------------------------------
## Preprocessing steps for SSM
probedDataDirs = sorted(glob(osp.join(dataDir, '*')))
num_subjects = len(probedDataDirs)
i = 5
VEL = []
sub_dict_list = []
print('Preprocessing dataset...')
for i in tqdm(range(num_subjects)):

    pid = osp.basename(probedDataDirs[i])
    os.makedirs(osp.join(outDir, 'clinical_cohort', pid), exist_ok=True)

    # read input data
    input_vtps = [pv.read(i) for i in sorted(glob(osp.join(probedDataDirs[i], '*.vtp')))]
    num_frames = len(input_vtps)

    # adjust units to m and m/s
    input_vtps = ut.adjust_units(input_vtps, array_name='Velocity')

    # landmark for in-plane rotation: point with max z coordinate (most left in patient's direction). doublecheck for all patients
    lm_ids = [np.argmax(input_vtps[k].points[:, 2]) for k in range(num_frames)]

    # create fixed plane points
    fxdpts, fxd_lm = ut.set_fixed_points(r_spac=0.05, circ_spac=5)

    # -------------------------- alignment
    # 0. check normals
    normals = [input_vtps[k].compute_normals()['Normals'] for k in range(num_frames)]
    signs = np.dot(input_vtps[3]['Velocity'], normals[3].mean(0))
    normals = [normals[k] * -1 for k in range(num_frames)]

    # 1.center at origin
    coms = [np.mean(np.array(input_vtps[k].points), 0) for k in range(num_frames)]
    xyz = [input_vtps[k].points - coms[k] for k in range(num_frames)]

    # 2. rotate s.t. normal = [0, 0, 1]
    new_normal = np.asarray([0, 0, 1])
    Rots = [ut.rotation_matrix_from_vectors(normals[k].mean(0), new_normal) for k in range(num_frames)]
    pts = [Rots[k].dot(xyz[k].T).T for k in range(num_frames)]
    for k in range(num_frames): pts[k][:, -1] = 0.
    vel = [Rots[k].dot(input_vtps[k]['Velocity'].T).T for k in range(num_frames)]

    # 3. normalize w.r.t. max coordinate norm
    Xmax = [np.max(np.sqrt(np.sum(xyz[k] ** 2, axis=1))) for k in range(num_frames)]
    pts = [pts[k] / Xmax[k] for k in range(num_frames)]

    # 4. second rotation to ensure consistent in-plane alignment
    Rots_final = [ut.rotation_matrix_from_vectors(pts[k][lm_ids[k], :], fxd_lm) for k in range(num_frames)]
    pts = [Rots_final[k].dot(pts[k].T).T for k in range(num_frames)]
    vel = [Rots_final[k].dot(vel[k].T).T for k in range(num_frames)]

    # create new polydatas
    aligned_planes = [input_vtps[k].copy() for k in range(num_frames)]
    for k in range(num_frames):
        aligned_planes[k].points = pts[k]
        aligned_planes[k]['Velocity'] = vel[k]

    # -------------------------- spatial interpolation
    interp_planes = ut.interpolate_profiles(aligned_planes, fxdpts, intp_options)

    # -------------------------- temporal interpolation
    tinterp_planes = ut.time_interpolation(interp_planes, time_intp_options)
    flowRate = ut.compute_flowrate(tinterp_planes)['Q(t)']
    peak = np.argmax(np.abs(flowRate))
    q = deque(np.arange(len(tinterp_planes)))
    circshift = 3 - peak
    q.rotate(circshift)
    tinterp_planes = [tinterp_planes[k] for k in q]

    for k in range(len(tinterp_planes)):
        tinterp_planes[k].save(osp.join(outDir, 'clinical_cohort', pid, 'prof_{:02d}.vtp'.format(k)))

    VEL.append([tinterp_planes[k]['Velocity'] for k in range(len(tinterp_planes))])

VEL = np.array(VEL)



#-----------------------------------------------------------------------------------------------------------------------
## Assemble and save matrix V
n_pat, n_frames, n_nodes = np.shape(VEL)[:-1]
V = np.empty((n_pat, n_frames*n_nodes*3))
for i in range(n_pat):
    uvw = VEL[i].flatten()
    V[i, :] = uvw        # PCA requires both variables and observations to be assembled into a single column vector

feat_cols = ['vel'+str(i) for i in range(V.shape[1])]
df = pd.DataFrame(V, columns=feat_cols)
df.to_csv(osp.join(outDir, 'matrixV.csv'), index=False)


#-----------------------------------------------------------------------------------------------------------------------
## PCA
V_mean = V.mean(0).reshape(n_frames, n_nodes, 3)

# Compute individual and cumulative variance
pca = PCA(n_components=18)                             # total n. of components has to be equal to min(n_samples, n_variables)
pca.fit(V)
var_components = pca.explained_variance_ratio_         # individual variance assciated to each component (mode)
var = np.sum(pca.explained_variance_ratio_[:26])       # total variance

cum_explained_var = []                                 # cumulative variance: # sum of invidual variances for subsequent components up to 10th mode.

for i in range(0, len(pca.explained_variance_ratio_)):
    if i == 0:
        cum_explained_var.append(pca.explained_variance_ratio_[i])
    else:
        cum_explained_var.append(pca.explained_variance_ratio_[i] + cum_explained_var[i-1])
cum_explained_var = np.asarray(cum_explained_var)


#-----------------------------------------------------------------------------------------------------------------------
# Shape Sampling: synthetic dataset generation
mean_profs = [pv.read(i) for i in sorted(osp.join('data', 'mean_profile', '*.vtp'))]
a = pca.components_.T
lam = pca.explained_variance_

M = 18
valid_count = 0
synth_ds = []
for i in tqdm(range(500)):
    variation = 0
    for m in range(M):
        c = random.uniform(-1.5, 1.5)
        variation += c * np.sqrt(lam[m]) * a[:, m]
    U = (V_mean + variation).reshape((n_frames, n_nodes, 3))
    new_profs = [mean_profs[0].copy() for _ in range(n_frames)]
    synthOutDir = osp.join(outDir, '{:03d}'.format(valid_count))
    os.makedirs(synthOutDir, exist_ok=True)
    for k in range(len(new_profs)):
        new_profs[k]['Velocity'] = U[k]
        new_profs[k].save(osp.join(synthOutDir, '{:03d}_{:02d}.vtp'.format(valid_count, k)))
    valid_count += 1




