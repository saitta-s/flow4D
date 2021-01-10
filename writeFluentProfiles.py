from os.path import join
from glob import glob
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt


## Settings
probedDataDir   = ''
profDir         = ''
showPlot        = False

t    = np.arange(0, 1.08, 0.001)  # simulation time steps
t4df = np.linspace(0, 1.08, 21)   # 4D flow frames

##
csvFiles = sorted(glob(join(probedDataDir, '*.csv')))
npts = len(pd.read_csv(csvFiles[0])['x'].tolist())

u = np.zeros((npts, len(t4df)))
v = np.zeros((npts, len(t4df)))
w = np.zeros((npts, len(t4df)))

i = 0
for f in csvFiles:
    df = pd.read_csv(f)
    npts = df.shape[0]
    xx = df['x'].tolist()
    yy = df['y'].tolist()
    zz = df['z'].tolist()
    u[:, i] = np.array(df['u']) * 10    # double check 10 factor
    v[:, i] = np.array(df['v']) * 10    # double check 10 factor
    w[:, i] = np.array(df['w']) * 10    # double check 10 factor
    i += 1


fu = np.zeros((npts, len(t)))
fv = np.zeros((npts, len(t)))
fw = np.zeros((npts, len(t)))
for p in range(npts):
    fu[p, :] = interpolate.interp1d(t4df, u[p, :], kind='cubic')(t)
    fv[p, :] = interpolate.interp1d(t4df, v[p, :], kind='cubic')(t)
    fw[p, :] = interpolate.interp1d(t4df, w[p, :], kind='cubic')(t)


if showPlot:
    f1 = plt.figure()
    ax1 = f1.add_subplot(111)
    ax1.plot(t4df, np.mean(u, axis=0), color='green', ls=':')
    ax1.plot(t4df, np.mean(v, axis=0), color='blue', ls=':')
    ax1.plot(t4df, np.mean(w, axis=0), color='red', ls=':')
    ax1.plot(t, np.mean(fu, axis=0), color='green', label='u')
    ax1.plot(t, np.mean(fv, axis=0), color='blue', label='v')
    ax1.plot(t, np.mean(fw, axis=0), color='red', label = 'w')
    plt.legend()


for i in range(len(t)):
    with open(join(profDir,'profile-{:04d}.prof'.format(i)), 'w') as fn:
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
        for ui in fu[:,i]:
            fn.write(str(ui) + '\n')
        fn.write(')\n')
        fn.write('(v\n')
        for vi in fv[:,i]:
            fn.write(str(vi) + '\n')
        fn.write(')\n')
        fn.write('(w\n')
        for wi in fw[:,i]:
            fn.write(str(wi) + '\n')
        fn.write(')\n')
        fn.write(')')

