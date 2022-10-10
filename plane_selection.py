

import os
import os.path as osp
from glob import glob
import numpy as np
import pyvista as pv
from vmtk import vmtkscripts

import utils as ut


#-----------------------------------------------------------------------------------------------------------------------
## Options
outputDir = r'' # path for saving probed .vtp files
saveName = ''   # filename of resamples .vtp files
source_flow_dir = r'' # directory containing .vtk files of a 4D flow acquisition (processed by dicoms_to_vtk.py)
source_mask_fn = r'' # path of the .vtk file containing the binary segmentation mask: it must be aligned with .vtk 4D flow files


#-----------------------------------------------------------------------------------------------------------------------
## Read data
flowData = [pv.read(fn) for fn in sorted(glob(osp.join(source_flow_dir, '*.vtk')))]
mask = pv.read(source_mask_fn)


#-----------------------------------------------------------------------------------------------------------------------
## Extract centerline
seg = mask.contour(1, method='marching_cubes', rng=[0.5, 1.5])
surf = seg.smooth(n_iter=1000, relaxation_factor=0.01)
surf = pv.wrap(ut.fillHoles(surf, holeSize=40))
cl = pv.wrap(ut.extract_parent_centerline(surf, dx=0.0001, smoothing_iters=50, smoothing_factor=0.5))


#-----------------------------------------------------------------------------------------------------------------------
## Select plane for probing 4D flow data

class MyCustomRoutine:
    def __init__(self, mesh, surface, centerline, flow):
        self.output = mesh  # Expected PyVista mesh type
        # default parameters
        self.kwargs = {
            'idx': -50,
            'frame': 3
        }
        self.surface = surface
        self.centerline = centerline
        self.flow = flow

    def __call__(self, param, value):
        self.kwargs[param] = value
        self.update()

    def update(self):
        idx = self.kwargs['idx']
        frame = self.kwargs['frame']
        new_plane = pv.Plane(center=self.centerline.points[idx],
                             direction=self.centerline['FrenetTangent'][idx],
                             i_size=0.1, j_size=0.1, i_resolution=200, j_resolution=200)
        probed = self.flow[frame].probe(new_plane)
        result = probed
        self.output.overwrite(result)
        p.update_scalar_bar_range([0, 0.4])
        return


p = pv.Plotter()
p.add_mesh(surf, color='white', opacity=0.3, pickable=False)
p.add_mesh(cl, color='black', line_width=5, pickable=True)
p.add_axes()
idx_0 = -50
starting_mesh = pv.Plane(center=cl.points[idx_0], direction=cl['FrenetTangent'][idx_0], i_size=0.1, j_size=0.1, i_resolution=200, j_resolution=200)
starting_mesh['Velocity'] = np.zeros((starting_mesh.number_of_points, 3))
p.add_mesh(starting_mesh, scalars='Velocity', opacity=0.9, show_edges=False, clim=[0, 0.7])
engine = MyCustomRoutine(starting_mesh, surf, cl, flowData)

p.add_slider_widget(
    callback=lambda value: engine('idx', int(value)),
    rng=[0, cl.number_of_points],
    value=cl.number_of_points//2,
    title="Point ID",
    pointa=(.67, .6), pointb=(.98, .6),
    style='modern',
)

p.add_slider_widget(
    callback=lambda value: engine('frame', int(value)),
    rng=[0, len(flowData)],
    value=3,
    title="Cardiac frame",
    pointa=(.67, .4), pointb=(.98, .4),
    style='modern',
)

p.show()

probed = engine.output
probed.compute_implicit_distance(surf, inplace=True)
result_plane = probed.threshold(0.0, scalars="implicit_distance", invert=True).extract_surface()


#-----------------------------------------------------------------------------------------------------------------------
## Probe 4D flow data and save
os.makedirs(outputDir, exist_ok=True)
probedDir = osp.join(outputDir, 'probed_planes')
for k in range(len(flowData)):
    probed_plane = flowData[k].probe(result_plane)
    probed_plane.save(osp.join(outputDir, saveName + '_probed_{:02d}.vtp'.format(k)))

cl.save(osp.join(outputDir, 'centerline.vtp'))
surf.save(osp.join(outputDir, 'surface.vtp'))
