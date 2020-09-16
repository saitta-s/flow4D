import os
from os.path import join
import sys
from tqdm import tqdm
import pydicom
from glob import glob
import re

def readDicomFolder(folder):
    series = []
    files = sorted(glob(join(folder, '*')))
    for file in tqdm(files, desc='Reading images', disable=len(files) == 0):
        ds = pydicom.dcmread(join(file))
        dataTemp = dict()
        dataTemp['SeriesNumber']    = int(ds.SeriesNumber)
        dataTemp['SliceLocation']   = float(ds.SliceLocation)
        dataTemp['FileName']        = file
        dataTemp['pixel_array']     = ds.pixel_array.astype('float')
        series.append(dataTemp)
    return ds, series

class flowReader(object):
    def __init__(self):
        self.ds      = None
        self.series0 = []
        self.series1 = []
        self.series2 = []
        self.series3 = []
        self.FOLDER_BASED_FLAG = False
        self.minusX = False
        self.minusY = False
        self.minusZ = False

    def read(self, datadir, method='folders'):
        if method == 'folders':
            self.readFolderBased(datadir)
        elif method == 'series':
            self.readSeriesBased(datadir)
        else:
            print('Specify reading method between folders and series. Exiting.')
            sys.exit()


    def readFolderBased(self, datadir):
        print('Reading folders')
        dirs = glob(join(datadir, '*'))
        for d in dirs:
            if re.search('mag|anatomy', os.path.basename(d), re.IGNORECASE):
                ds, series0 = readDicomFolder(d)
            elif re.search('fh|hf|si|is', os.path.basename(d), re.IGNORECASE):
                _, series1 = readDicomFolder(d)
                if re.search('fh|is', os.path.basename(d), re.IGNORECASE):
                    self.minusX = True
            elif re.search('ap|pa', os.path.basename(d), re.IGNORECASE):
                _, series2 = readDicomFolder(d)
                if re.search('pa', os.path.basename(d), re.IGNORECASE):
                    self.minusY = True
            elif re.search('rl|lr', os.path.basename(d), re.IGNORECASE):
                _, series3 = readDicomFolder(d)
                if re.search('lr', os.path.basename(d), re.IGNORECASE):
                    self.minusZ = True
            else:
                print('Error in naming the folders. Exiting.')
                sys.exit()

        self.series0 = series0
        self.series1 = series1
        self.series2 = series2
        self.series3 = series3
        self.ds = ds
        self.FOLDER_BASED_FLAG = True


    def readSeriesBased(self, datadir):
        print('Reading folders.')
        series0 = []
        series1 = []
        series2 = []
        series3 = []
        series = []
        sNum = []
        for root, dirs, files in os.walk(datadir):
             for file in tqdm(files, desc='Reading images', disable=len(files)==0):
                 ds = pydicom.dcmread(join(root, file))
                 sNum.append(ds.SeriesNumber)
                 dataTemp = dict()
                 dataTemp['SeriesNumber'] = int(ds.SeriesNumber)
                 dataTemp['SliceLocation'] = float(ds.SliceLocation)
                 dataTemp['FileName'] = file
                 dataTemp['pixel_array'] = ds.pixel_array.astype('float')
                 series.append(dataTemp)

        sNum = sorted(set(sNum))

        for i in range(len(series)):
            if series[i]['SeriesNumber'] == sNum[0]:
                series0.append(series[i])
            elif series[i]['SeriesNumber'] == sNum[1]:
                series1.append(series[i])
            elif series[i]['SeriesNumber'] == sNum[2]:
                series2.append(series[i])
            elif series[i]['SeriesNumber'] == sNum[3]:
                series3.append(series[i])

        self.series0 = series0
        self.series1 = series1
        self.series2 = series2
        self.series3 = series3
        self.ds = ds
        self.FOLDER_BASED_FLAG = False



