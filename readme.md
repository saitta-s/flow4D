# flow4D
This repository can be used to work with 4D flow MRI acquisitions for CFD applications. <br/>

## Features
The main provided functionalities enable to: 
- read 4D flow dicom files and generate .vtk files that can be visualized in Paraview;
- extract velocity profiles from 4D flow and map them to the inlet of a target model;
- write boundary conditions files for multiple CFD software, including Fluent, CFX, Star-CCM+, OpenFoam, SimVascular
- generate an arbitrary number of realistic velocity profiles from a statistical shape model

## Installation
To run the scripts you need a python interpreter. The use of a conda environment is 
strongly recommended. 

### Dependencies
- [vmtk](https://github.com/conda-forge/vmtk-feedstock)
- pyvista


## Read and process 4D flow dicom files
```
python dicoms_to_vtk.py
```

## Interactive plane selection and velocity profile extraction
```
python plane_selection.py
```

## Map precomputed velocity profiles to a target inlet shape
```
python mapping.py
```


## Reading method
Two methods for reading 4D flow acquisitions were implemented.<br/>
1. 'folders': to use if you pre-organized the 4D flow images in 4 separate folders, one containing the string 'mag' 
for the magnitude sequence, and 3 named by the phase thay encode for. For example, the phase encoding the foot-head 
direction will be named either 'fh' or 'hf'. Careful, folders' names are direction-specific, meaning that if the velocity
direction is positive from head to feet, the folder must be named 'hf'.
2. 'series': to use only if the magnitude and the 3 phases sequences all have different series names. You can check this
by opening the parent folder containing all dicoms with Radiant. If Radiant will detect 4 separate acquisitions (in the
left column), then you may use this method without worrying about manually naming the 4 subfolders. 
This method has only been tested for Siemens, Philips and GE data.

## Settings
Set the correct paths in the 'preferences' section at the beginning of the script 'readAndProcessDicoms.py'<br/>
* datadir = full path to the folder containing 4D flow dicom data.
* units = string containing the measurement units of the output files. ("m/s" recommended).
* saveFormat = string containing the format of the output files. (".vtk" recommended)
* outdir = full path to the folder where you want to save the processed files.
* saveName = string containing the name of the folder where you want to save the processed files. It will be created inside outdir.
* prefix = string containing the name of the files that will be written.
* venc = list of length 3 containing VENC values in cm/s. Example: [150, 150, 150]
* reading_method = string containin the chosed reading method. Choice between 'series' and 'folders'.


## Run the code
To run the script 'readAndProcessDicoms.py', make sure you set the preferences correctly, then open cmd or powershell and follow these steps:
```
cd flow4D
python readAndProcessDicoms.py
```

## Visualize processed data
Open Paraview and follow these steps:
1. File -> Open
2. Browse to the written .vtk files.
3. Select the series (paraview) should automatically recognize the files as a series.
4. Select the reader calles 'XML Structured Grid Reader' or 'Structured Grid Reader' depending on your version of Paraview.
5. Double check the velocity directions.
6. Enjoy!
