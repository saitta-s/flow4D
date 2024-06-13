from pathlib import Path
import os
from os.path import join

inputDir = '/Users/scarpma/Downloads/Kowal_RM/S2RAQXBQ/'
outputDir = '/Users/scarpma/Downloads/Kowal_RM/S2RAQXBQ/flow'
exclude = ['VERSION', '.DS_Store']
if not os.path.exists(outputDir): 
  os.mkdir(outputDir)

for ii, (root, dirs, files) in enumerate(os.walk(inputDir)):
  files = [file for file in files if file not in exclude]
  if len(files)==1500:
    print(ii, root)
    for file in files:
      outputFilename = str(ii)+'_'+file+'.dcm'
      #print(join(outputDir,outputFilename))
      os.symlink(join(root,file), join(outputDir,outputFilename))
