#!/usr/bin/env python3
import numpy as np
import math as m
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
from scipy.signal import argrelmax, argrelmin
import sys
import os
import argparse
from sklearn.metrics import mean_squared_error


###### Functions #######


#### load in the dat file as a dataframe 

def load_data(i):
	itfrmt =  str(i).zfill(2)
	path2data = 'it'+str(itfrmt)+'/analysis/'
	data = pd.read_csv(path2data+dat, delim_whitespace=True, header=None)
	#dataf8 = np.loadtxt(path2data+dat, usecols=(range(rc+3)),delimiter=None, dtype=('f8,f8,f8,f8,f8'))
	return data

#### load the slurmout file for equilibration

def load_analysis(i):
	itfrmt =  str(i).zfill(2)
	path2data = 'it'+str(itfrmt)+'/'
	
	def filter_teq(line):
		return 'Teq' in line

	skip = [i for i,line in enumerate(open(path2data+eq)) if not filter_teq(line)]
	analysis = pd.read_csv(path2data+eq, delim_whitespace=True, skip_blank_lines=True, header=None, skiprows=skip, usecols=range(19))

	return analysis

#### calculate the free energy barrier

def get_barrier(fe):
	maximum = fe.max()
	max_idx = fe.argmax()
	beforemax = fe[0:max_idx]
	minimum = beforemax.min()
	G = maximum - minimum
	return G

### Calculate the sum of euclidean distances of points from two curves based on index. Column number will be determined from number of reaction coordinates

def euclid_byindex(df1, df2, cols):
	return np.sum(np.linalg.norm(df1[cols].values - df2[cols].values, axis=1))

### Calculate the sum of euclidean distances between two free energy profiles

def euclid_byindex_fe(df1, df2, col):
	df1[col] = df1[col] - df1[col].iloc[0]
	df2[col] = df2[col] - df2[col].iloc[0]
	return np.sum(np.linalg.norm(df1[col].values - df2[col].values))

### Calculate the sum of euclidean distances of points from two curves based on the closest point. Column number will be determined from number of reaction coordinates

def euclid_bynearest(df1, df2, cols):
	alldist = []
	for i in df1[cols].values:
		dists = []
		for j in df2[cols].values:
			dists.append(np.linalg.norm(i - j))		
		mindist = np.array(dists).min()
		alldist.append(mindist)
	return np.sum(alldist)	

### Calculate the RMSD between the last 5 values in an array

def RMSD_last5(a):
	a5 = a[-6:-1]
	mean = np.mean(a5)
	return np.sqrt(mean_squared_error(a5[-1], mean)/5)

### Calculate the RMSD between the last N values in an array

def RMS_lastN(N, a):
	aN = a[-N:-1]
	#RMSN = np.sqrt(np.sum(np.square(aN)+np.square(a[-1]))/len(aN))
	RMSN = np.sqrt(np.sum(np.square(np.mean(aN))+np.square(a[-1]))/2)
	#RMSN = np.sqrt(np.sum(np.square(aN))/len(aN))
	return RMSN

### Calculate the sqrt of the varience of the last N values in an array

def sqrt_varN(N, a):
	aN = a[-N:-1]
	mean = np.mean(aN)
	sqvar = np.sqrt(np.sum(np.square(aN - mean))/len(aN))
	return sqvar

###### Arugments #######

if __name__ == "__main__":

	parser = argparse.ArgumentParser \
	( formatter_class=argparse.RawDescriptionHelpFormatter,description="Test strings for convergence")

	parser.add_argument \
	        ("-i","--iter",
	         help="last iteration to be considered in analysis",
	         type=int,
	         required=True )

	parser.add_argument \
	        ("-r","--rc",
	         help="number of reaction coordinates (also the number of columns in p.dat containing path info)",
	         type=int,
	         required=True )

	parser.add_argument \
			("-d","--dat",
	         help="name of dat file created with ndfes-PlotMultistringPath.py (should contain path and free energies)",
	         type=str,
	         required=True )

	parser.add_argument \
			("-p","--pmf",
			 action='store_true',
			 help="Use this flag if you want to analyze the convergence of the PMF",
			 required=False)

	parser.add_argument \
			("-e","--eq",
			 help="If you want to analyze equilibration, what is the name of the analysis output file? It is likely analysis.slurmout",
			 type=str,
			 required=False)


	args = parser.parse_args()

	it = args.iter
	rc = args.rc 
	dat = args.dat
	eq = args.eq

	#### In the dat file, the first column is profress, then reaction coordinates. The free energy is in the column after the last reaction coordinate. 

	fecol = rc+1 
	rc_cols = [i+1 for i in range(rc)]
	its = [ i + 1 for i in range(it)]

	iteration, err_path_seq, err_path_near, err_pmf, err_barrier, Gddag, fractprod = [], [], [], [], [], [], []
	
	N = [2, 3, 4, 5]

	RMS_pmf = [[] for _ in range(len(N))]
	RMS_path_seq = [[] for _ in range(len(N))]
	RMS_path_near = [[] for _ in range(len(N))]


##### compare string iterations

	for i in range(it):
		
		current = i+1
		
		datacurrent = load_data(current)

		#### compute the barrier ####

		if args.pmf:
			Gddag.append(get_barrier(datacurrent.iloc[:, fecol]))

		if args.eq:

			outfile = load_analysis(i=current)
			Ninp = np.array(outfile[2].sum())
			Nout = np.array(outfile[4].sum())
			fractprod.append(np.divide(Nout, Ninp))
	
		### compare the current to the previous iteration unless this is the first iteration

		if i != 0:
			iteration.append(current)
			last = i
			datalast = load_data(last)

			#### get sum of squared distance vector by index for path ####

			
			euclid_dist_index = euclid_byindex(datacurrent, datalast, rc_cols)
			err_path_seq.append(euclid_dist_index)

			#### get sum of squared distance vector by nearest neighbor for path ##

			err_path_near.append(euclid_bynearest(datacurrent, datalast, rc_cols))
		
			#### get sum of squared distance vector by index for pmf ####
			if args.pmf:
				err_pmf.append(euclid_byindex_fe(datacurrent, datalast, fecol))


		for n in range(len(N)):
			if current >= N[n]+1:
				if args.pmf:
					RMS_pmf[n].append(RMS_lastN(N=N[n], a=err_pmf))
				RMS_path_seq[n].append(RMS_lastN(N=N[n], a=err_path_seq))
				RMS_path_near[n].append(RMS_lastN(N=N[n], a=err_path_near))


	if it >= 5:
		if args.pmf:
			print('Barrier for last 5 iterations:',np.around(Gddag[-6:-1],2),' kcal/mol')

	else:
		if args.pmf:
			print('Barrier for ',it,' iterations:', np.around(Gddag),' kcal/mol')


	if args.eq:
		print('Average percentage of production simulation:',np.around(np.mean(fractprod),2))
		#print(RMS_path_near)

	##### plotting related crap ######


	### plot path convergence ##

	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()

	v1 = ax1.plot(iteration, err_path_seq, color='green', marker='s', markersize=5, label=r'$\Sigma$L2(Path by index)')
	v2 = ax1.plot(iteration, err_path_near, color='red', marker='s', markersize=5, label=r'$\Sigma$L2(Path by nearest)')
	


	ax1.set_xlabel('String iteration', fontsize=14)
	ax1.set_ylabel(r'$\Sigma$(L2)', fontsize=14)
	ax1.xaxis.set_tick_params(labelsize=12)
	ax1.yaxis.set_tick_params(labelsize=12)

	if args.pmf:
		v3 = ax2.plot(its, Gddag, color='blue', marker='o', markersize=5, label=r'$\Delta$G$^{\ddag}$' + " " + '(kcal/mol)')
		v4 = ax1.plot(iteration, err_pmf, color='black', marker='s', markersize=5, label=r'$\Sigma$L2(Free energy profile)')
		ax2.set_ylabel(r'$\Delta$G$^{\ddag}$' + " " + '(kcal/mol)', fontsize=14, color='blue')
		ax2.yaxis.set_tick_params(labelsize=12)
		v = v1+v2+v3+v4
	else:
		v = v1+v2

	labs = [i.get_label() for i in v]
	ax1.legend(v, labs)

	#pngname = 'convergence',it,'.png'

	fig.tight_layout()
	plt.savefig('convergence_test.png', dpi=300)

	##### plot RMS #####

	if args.pmf:
		fig, ax = plt.subplots(3, sharex=True, figsize=(7,13))
	else: 
		fig, ax = plt.subplots(2, sharex=True, figsize=(7,13))

	for i in range(len(N)):
		#itsforplot = [j + 1 for j in range(len(RMS_path_near[i]))]
		prev = it - N[i]
		itsforplot = iteration[-prev:]

		print('N=',N[i])
		ax[0].plot(itsforplot, RMS_path_seq[i], marker='o', markersize=4, label='Average over last {}'.format(N[i]))
		
		print('RMS(path by index)',np.around(RMS_path_seq[i][-5:],2))
		ax[1].plot(itsforplot, RMS_path_near[i], marker='o', markersize=4, label='Average over last {}'.format(N[i]))
		
		print('RMS(path by nearest)',np.around(RMS_path_near[i][-5:],2))
		
		
		if args.pmf:
			ax[2].plot(itsforplot, RMS_pmf[i], marker='o', markersize=4, label='Average over last {}'.format(N[i]))
			print('RMS(pmf)',np.around(RMS_pmf[i][-5:],2))
			print('')
		
	ax[0].plot(iteration, err_path_seq, color='black', marker='s', markersize=5, label=r'$\Sigma$L2(Path by index)')
	ax[1].plot(iteration, err_path_near, marker='s', markersize=5, color='black', label=r'$\Sigma$L2(Path by nearest)')
	
	if args.pmf:	
		ax[2].plot(iteration, err_pmf, color='black', marker='s', markersize=5, label=r'$\Sigma$L2(Free energy profile)')
		ax[2].xaxis.set_tick_params(labelsize=12)
		ax[2].yaxis.set_tick_params(labelsize=12)
		ax[2].set_xlabel('String iteration', fontsize=12)
		ax[2].set_ylabel('RMS(PMF)', fontsize=14)
		ax[2].legend(fontsize=11)


	ax[0].xaxis.set_tick_params(labelsize=12)
	ax[0].yaxis.set_tick_params(labelsize=12)
	ax[1].xaxis.set_tick_params(labelsize=12)
	ax[1].yaxis.set_tick_params(labelsize=12)
	ax[0].set_ylabel('RMS(Path by index)', fontsize=14)
	ax[1].set_ylabel('RMS(Path by nearest)', fontsize=14)
	ax[0].legend(fontsize=11)
	ax[1].legend(fontsize=11)
	
	plt.xticks(its)
	fig.tight_layout()
	plt.savefig('RMS_lastN_test.png', dpi=300)
	plt.show()



	#### plot equilibration ####
	if args.eq:
		fig, ax = plt.subplots()

		barx = [j + 1 for j in range(it)]
		bareq = [1 - k for k in fractprod]

		ax.bar(barx, bareq, color='red')
		ax.bar(barx, fractprod, color='green', bottom = bareq)
		ax.set_xlabel('String iteration', fontsize=14)
		ax.set_ylabel('Percent equilibrated', fontsize=14)
		ax.xaxis.set_tick_params(labelsize=12)
		ax.yaxis.set_tick_params(labelsize=12)	
		plt.xticks(barx)

		plt.savefig('prod_eq_bar_test.png', dpi=300)
		plt.show()
















			


			

	
	



		






		







