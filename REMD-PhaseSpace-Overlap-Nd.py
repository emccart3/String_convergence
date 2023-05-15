#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import scipy.stats as stats
import math as m
from statistics import NormalDist
import pytraj as pt
from pathlib import Path
import os
from glob import glob


#### Change these ##### 
nwin = 32
lims = [[-1.2, 1.3], [-2.2, 2.2]]  ### Define the space your reaction coordinates approximately fall in 
rc_mask = [[':9@N3 :9@H3', ':69@N3 :9@H3'], [':69@O :69@C5', ':62@N1 :69@C5']]  #### distance/angles in the reaction coordinates (multiple entries in the sublist is a difference of distance)
trajdir = 'it25' ## relative path to directory containing trajetories
parm = 'template/qmmm.parm7'
color = ['red', 'blue']  ### colors for your plots so matplotlib doesn't pick weird ones

#### Check to make sure the given mask and limits are the same length

if len(lims) != len(rc_mask):
    print('Length of rc_mask and lims must be the same')
    exit(1)

if len(lims) != len(color):
    print('Length of color and lims must be the same')
    exit(1)

##### Determine what kind of reaction coordinates you have from the mask based on the spaces

def parse_rc_mask(rc_mask):
    coordtypes = []
    numbcoords = len(rc_mask)
    print('There are {} reaction coordinates'.format(numbcoords))
    for i in range(len(rc_mask)):
        if len(rc_mask[i]) == 1:
            count = 0
            for j in rc_mask[i]:
                for k in j:
                    if k == " ":
                        count += 1
            if count == 1:
                print('Reaction coordinate {} is a distance'.format(i+1))
                coordtypes.append('d')
            elif count == 2:
                print('Reaction coordinate {} is an angle'.format(i+1))
                coordtypes.append('a')
        elif len(rc_mask[i]) == 2:
            print('Reaction coordinate {} is a difference of distances'.format(i+1))
            coordtypes.append('dd')
    return(coordtypes)

#### get the average value and standard deviation of the reaction coordinates

def meanrc(rctype, mask, traj):
    if rctype == 'dd':
        mean = np.mean(pt.distance(traj, mask[0]) - pt.distance(traj, mask[1]))
        stdev = np.std(pt.distance(traj, mask[0]) - pt.distance(traj, mask[1]))
    elif rctype == 'd':
        mean = np.mean(pt.distance(traj, mask[0]))
        stdev = np.std(np.mean(pt.distance(traj, mask[0])))
    elif rctype == 'a':
        mean = np.mean(pt.angle(traj, mask[0]))
        stdev = np.std(pt.angle(traj, mask[0]))
    return(mean, stdev)


fig, ax = plt.subplots(ncols=2, nrows=len(lims), sharex=False, figsize=(7, 4))

### get the coordinate types and make sure they match the length of the rc_mask

coordtypes = parse_rc_mask(rc_mask)
#print(coordtypes)

if len(coordtypes) != len(rc_mask):
    print('Coordtypes could not be parsed correctly')
    exit(1)
else:
    print('Length of Coordtypes and rc_mask are the same')


win = [i for i in range(1, nwin+1)]
progress = np.linspace(0, 1, nwin)
trajfilesraw = list(glob(str(trajdir+'/'+'img*.nc')))

trajfiles = sorted(trajfilesraw, key=lambda x: int(x.split('/')[-1].split('img')[1].split('.nc')[0].zfill(2)))

#print(trajfiles)

#### set up and array of points between the given limits 

space = []

for i in range(len(lims)):
    j = round((lims[i][1] - lims[i][0])*100)
    print(j)
    space.append(np.linspace(lims[i][0], lims[i][1], j))


### Calculate the means and standard deviations

muRC = [[] for _ in range(len(lims))]
stdevRC = [[] for _ in range(len(lims))]
overlap = [[] for _ in range(len(lims))]

for t in range(len(trajfiles)):
    traj = pt.iterload(trajfiles[t], parm)
    for i in range(len(lims)):
        mean, stdev = meanrc(coordtypes[i], rc_mask[i], traj)
        muRC[i].append(mean)
        stdevRC[i].append(stdev)
        ### construct gaussians from mean and stdev
        norm = stats.norm.pdf(space[i], mean, stdev)
        ax[0, i].plot(space[i], norm, color=color[i])

        #### calculate the overlap between gaussians
        if t > 0:
            last = t - 1
            overlap[i].append(NormalDist(mu=mean, sigma=stdev).overlap(NormalDist(mu=muRC[i][last], sigma=stdevRC[i][last])))


for i in range(len(lims)):
    ax[1, i].plot(progress[0:-1], overlap[i], color=color[i], marker='o', markersize=4)
   # ax[i, 0].set_xlim([j for j in lims[i]])
    ax[1, i].set_xlim([-0.05,1.05])
    ax[1, i].set_xlabel('Progress', fontsize=12)
    ax[1, i].set_ylabel('Overlap', fontsize=12)
    ax[0, i].set_xlabel('RC{}'.format(i+1), fontsize=12)
    ax[0, i].set_ylabel('Probability Density'.format(i+1), fontsize=12)



plt.tight_layout()
plt.savefig('REMD_Overlap.png', dpi=300)

plt.show()











