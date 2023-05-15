#!/usr/bin/env python3
import numpy as np
import math as m
import pandas as pd
import pytraj as pt
import sys
import os
from scipy.spatial import distance

### Set system specific variables

parm = 'template/qmmm.parm7'
maxit = 25 ## number of string iterations 
nimg = 32  ## number of images per iteration
rc_mask = [[':9@N3 :9@H3', ':69@N3 :9@H3'], [':69@O :69@C5', ':62@N1 :69@C5']] ## This is the syntax pytraj takes
module = 'ndfes.module.1.8.1'

## Parse the rc_mask based on spaces

count = 0
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
        elif count == 2:
            print('Reaction coordinate {} is an angle'.format(i+1))
    elif len(rc_mask[i]) == 2:
            print('Reaction coordinate {} is a difference of distances'.format(i+1))

## Function for euclidean distance between n dimensional points x and y

def euclid_dist(x, y):
    dist = distance.euclidean(x, y)
    return(dist)

### Create a list with the trajectory file paths 

def get_file_paths(maxit):
    files = []
    paths = []
    for it in range(1, maxit+1):
        iteration = 'it'+str(it).zfill(2)
        for img in range(1, nimg+1):
            image = 'img'+str(img).zfill(2)+'.nc'
            imagepath = 'img'+str(img).zfill(2)
            ncpath = iteration+'/'+image
            path = iteration+'/'+imagepath
            files.append(ncpath)
            paths.append(path)
    pathsarr = np.array_split(np.array(paths),maxit)
    return(files, pathsarr)
    
## Determine what kind of reaction coordinates you have from the mask

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
coordtypes = parse_rc_mask(rc_mask)
print(coordtypes)

if len(coordtypes) != len(rc_mask):
    print('Coordtypes could not be parsed correctly')
    exit(1)
else:
    print('Length of Coordtypes and rc_mask are the same')

### Choose the pytraj command based on type of mask

def meanrc(rctype, mask):
    if rctype == 'dd':
        mean = np.mean(pt.distance(traj, mask[0]) - pt.distance(traj, mask[1]))
    elif rctype == 'd':
        mean = np.mean(pt.distance(traj, mask[0]))
    elif rctype == 'a':
        mean = np.mean(pt.angle(traj, mask[0]))
    return(mean)


files, paths = get_file_paths(maxit)

### Calculate the reaction coordinate averages from the trajectories using pytraj

muRC = [[] for _ in range(len(coordtypes))]
print('Getting mean values of reaction coordinates. Sit tight.')
for f in files:
    traj = pt.iterload(f, parm)
    for rc in range(len(rc_mask)):
        mean = meanrc(coordtypes[rc], rc_mask[rc])
        muRC[rc].append(mean)

### Load the string files and and format the values in a list

string_centers = [[] for _ in range(len(coordtypes))]
for i in range(maxit):
    itfrmt =  str(i+1).zfill(2)
    stringfile = 'string.'+itfrmt+'.dat'
    for j in range(len(coordtypes)):
        #string_centers[j].append(np.loadtxt(stringfile, usecols=[j]))
        if not os.path.exists(stringfile):
            print('{} is missing'.format(stringfile))
        else:
            strarray = np.loadtxt(stringfile, usecols=[j])
            for k in strarray:
                string_centers[j].append(k)

### Calculate euclidean distances between averages and string centers

distlst = []
for i in range(maxit*nimg):
    rcfromlst = list(list(zip(*muRC))[i])
    strfromlst = list(list(zip(*string_centers))[i])
    distlst.append(euclid_dist(x=rcfromlst, y=strfromlst))


### determine weights from euclidean distance  

maxdist = np.max(distlst)
print(maxdist)

weightslst = [round((d * 100)/maxdist) for d in distlst]
weights = np.array_split(np.array(weightslst),maxit)
print(len(weights[0]))


### run ndfes-Checkequil

rcinplst = [i+1 for i in range(len(rc_mask))]
sinp = str(rcinplst)[1:-1]
rcinp = sinp.replace(',', '')
os.system('module load {}'.format(module))
for i in range(len(paths)):
    newdir = 'it'+str(i+1).zfill(2)+'/prune_traj_weightbystring'
    if not os.path.exists(newdir):
        os.mkdir(newdir)
    for j in range(len(paths[i])):
        image = str(j+1).zfill(2)
        cmd = "ndfes-CheckEquil.py -p {0} -i {1}.nc ".format(parm, paths[i][j])+\
        "-o {0}/img{1}.nc -d {2}.disang -r {3} -m 0.75 --prune ".format(newdir, image, paths[i][j], rcinp)+\
        "--minsize {0} --maxsize {0}".format(weights[i][j])
        print('Running {}'.format(cmd))
        os.system(cmd)



