import sys
import os
import os.path as osp
import numpy as np
from glob import glob
import pyvista as pv

import utils as ut


#-----------------------------------------------------------------------------------------------------------------------
## Options
outputDir = r'' # path for saving resampled .vtp files
saveName = ''   # filename of resamples .vtp files
source_profile_dir = r'' # directory containing .vtp files associated to a 4D profile
target_profile_fn = r''  # can be a .stl, .vtk or .vtp file
flip_normals = True # usually set to True, but might have to change depending on target plane orientation
leftmost_idx_on_target = 000      # index of the leftmost point in the target plane w.r.t the subject
intp_options = {
    'zero_boundary_dist': 0.2, # percentage of border with zero velocity (smooth damping at the border)
    'zero_backflow': True, # set all backflow components to zero
    'kernel': 'linear', # RBF interpolation kernel (linear is recommended)
    'smoothing': 0.01, # interpolation smoothing, range recommended [0, 2]
    'degree': 0,
    'hard_noslip': False} # degree of polynomial added to the RBF interpolation matrix


#-----------------------------------------------------------------------------------------------------------------------
## Read data
source_profiles = [pv.read(i) for i in sorted(glob(osp.join(source_profile_dir, '*.vtp')))]
target_plane = pv.read(target_profile_fn)

num_frames = len(source_profiles)
source_pts = [source_profiles[k].points for k in range(num_frames)]
source_coms = [source_pts[k].mean(0) for k in range(num_frames)]
target_pts = target_plane.points
target_com = target_pts.mean(0)
target_normal = target_plane.compute_normals()['Normals'].mean(0)
normals = [source_profiles[k].compute_normals()['Normals'].mean(0) for k in range(num_frames)]
if flip_normals: normals = [normals[k] * -1 for k in range(num_frames)]


#-----------------------------------------------------------------------------------------------------------------------
## Align source to target

# center at origin for simplicity
target_pts -= target_com
source_pts = [source_pts[k] - source_coms[k] for k in range(num_frames)]

# normalize w.r.t. max coordinate norm
targetmax = np.max(np.sqrt(np.sum(target_pts ** 2, axis=1)))
pts = [source_pts[k] * targetmax for k in range(num_frames)]

# rotate to align normals
Rots = [ut.rotation_matrix_from_vectors(normals[k], target_normal) for k in range(num_frames)]
pts = [Rots[k].dot(pts[k].T).T for k in range(num_frames)]
vel = [Rots[k].dot(source_profiles[k]['Velocity'].T).T for k in range(num_frames)]

# second rotation to ensure consistent in-plane alignment
lm_ids = [np.argmax(source_pts[k][:, 0]) for k in range(num_frames)]
Rots_final = [ut.rotation_matrix_from_vectors(pts[k][lm_ids[k], :], target_pts[leftmost_idx_on_target, :]) for k in range(num_frames)]
pts = [Rots_final[k].dot(pts[k].T).T for k in range(num_frames)]
vel = [Rots_final[k].dot(vel[k].T).T for k in range(num_frames)]


# create new polydatas
aligned_planes = [source_profiles[k].copy() for k in range(num_frames)]
for k in range(num_frames):
    aligned_planes[k].points = pts[k]
    aligned_planes[k]['Velocity'] = vel[k]

# spatial interpolation
interp_planes = ut.interpolate_profiles(aligned_planes, target_pts, intp_options)
interp_planes[2].plot()

# recenter
for k in range(num_frames):
    interp_planes[k].points += target_com


#-----------------------------------------------------------------------------------------------------------------------
## Save profiles to .vtp
os.makedirs(outputDir, exist_ok=True)
for k in range(num_frames):
    interp_planes[k].save(osp.join(outputDir, saveName + '_{:02d}.vtp'.format(k)))


