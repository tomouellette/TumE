#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os
import argparse
import torch
import torch.nn.functional as F
from utils import *
from nets import *

"""
General script for processing and analyzing simulated data
"""

def arg_parse():
	parser = argparse.ArgumentParser()
	parser.add_argument('-a', '--analysis', type=str, help='The analysis task data is being parsed for')
	parser.add_argument('-d', '--dir', type=str, help='The directory where data is stored.')
	parser.add_argument('-o', '--out', type=str, help='The output directory')	
	args = parser.parse_args()
	return args

def metric_simulations(dir: str, out: str):
	"""
	Converts simulated data into readable format for applying R-based evolutionary analysis packages
	
	:param dir (str): Directory containing simulations
	:param out (str): Output directory to save files
	
	>return (np.ndarray) Each individual simulation saved as numpy vector containing [vaf, reads, depth] information
	"""
	for i in [j for j in os.listdir(dir) if 'vafdepth' in j]:
		data = np.load(dir + i, allow_pickle=True)
		for k in range(len(data)):
			vaf, depth = data[k][0], data[k][1]
			reads = np.array([int(i) for i in vaf * depth])
			arr = np.array([vaf, reads, depth])
			np.save(out + i + '_' + str(k) + '.npy' , arr)

def grab_synthetic_E_labels(dir: str, out: str, nm: str, count = False):
	""" 
	Get labels from simulated data

	:param dir (str): Directory containing simulations
	:param out (str): Output directory to save files
	:param nm (str): Name of output data frame in csv format
	
	>return (pd.DataFrame) Saved in CSV format with simulation labels
	"""
	dt = pd.DataFrame()
	counter = 0
	for k in [j for j in os.listdir(dir)]:
		counter += 1
		if count == True: print(counter)		
		data = np.load(dir + k, allow_pickle=True)
		features, mode, nsubclone, frequencies1, frequencies2, time1, time2 = sim2evo(data)
		depth = int(k.split('_')[1])
		rho = float(k.split('_')[2])
		identifier = [k.split('_features.npy')[0] + '_' + str(m) for m in range(len(data))] 
		add = pd.DataFrame({'identifier': identifier, 'depth':depth, 'rho':rho, 'mode': mode, 'nsub': nsubclone, 'scfreq1': frequencies1, 'scfreq2': frequencies2, 'sctime1':time1, 'sctime2':time2})
		dt = pd.concat([dt, add])
	dt.to_csv(out + nm, index = False)

def parse_synthetic_E_metrics(dir: str, out: str, nm: str):
	"""
	Merge all benchmark metrics csvs
	"""
	dt = pd.DataFrame()
	for k in os.listdir(dir):
		add = pd.read_csv(dir + k)
		dt = pd.concat([dt, add], ignore_index=True)
	dt.to_csv(out + nm)

def grab_synthetic_E_metric_labels(dir, out, nm, count = True):
	"""
	Grab properly ordered metric labels given each individual file was parsed in order of rows
	"""
	dt = pd.DataFrame()
	counter = 0
	for k in [j for j in os.listdir(dir)]:
		counter += 1
		if count == True: print(counter)		
		data = np.load(dir + k, allow_pickle=True)
		mode = np.array([i[0] for i in data[:,1]])
		nsub = np.array([i[1] for i in data[:,1]])
		#
		def format_scdata(fmt):
			fmt[np.where(np.array([type(i) for i in fmt]) == type(float()))] = [[i, float(0)] for i in fmt[np.where(np.array([type(i) for i in fmt]) == type(float()))]]
			fmt[np.where(np.array([len(i) for i in fmt]) == 1)] = [i + [float(0)] for i in fmt[np.where(np.array([len(i) for i in fmt]) == 1)]]		
			fmt = np.array([np.array(i) for i in fmt])			
			return fmt
		#
		freq = format_scdata(np.array([i[2] for i in data[:,1]]))
		time = format_scdata(np.array([i[3] for i in data[:,1]]))
		sfit = format_scdata(np.array([i[4] for i in data[:,1]]))
		sfit = np.array([sfit[i,np.argsort(freq, axis = 1)[i]] for i in range(len(freq))])
		time = np.array([time[i,np.argsort(freq, axis = 1)[i]] for i in range(len(freq))])
		freq = np.array([freq[i,np.argsort(freq, axis = 1)[i]] for i in range(len(freq))])
		identifier = [k.split('_features.npy')[0] + '_' + str(m) for m in range(len(data))] 
		add = pd.DataFrame({'identifier': identifier, 'mode': mode, 'nsub': nsub, 'scfreq1': freq[:,0], 'scfreq2': freq[:,1], 'sctime1':time[:,0], 'sctime2':time[:,1], 'sfit1': sfit[:,0], 'sfit2': sfit[:,1]})
		dt = pd.concat([dt, add])
	dt.to_csv(out + nm)

def grab_synthetic_F_labels(dir: str, out: str, nm: str, count = False):
	"""
	Get labels from simulated data

	:param dir (str): Directory containing simulations
	:param out (str): Output directory to save files
	:param nm (str): Name of output data frame in csv format
	
	>return (pd.DataFrame) Saved in CSV format with simulation labels
	"""
	dt = pd.DataFrame()
	counter = 0
	for k in [j for j in os.listdir(dir)]:
		counter += 1
		if count == True: print(counter)				
		data = np.load(dir + k, allow_pickle=True)
		features, mode, nsubclone, frequencies1, frequencies2, time1, time2 = sim2evo(data)
		birthrate = float(k.split('_')[1])
		deathrate = float(k.split('_')[2].split('.npy')[0])
		identifier = [k.split('_features.npy')[0] + '_' + str(m) for m in range(len(features))] 
		add = pd.DataFrame({'identifier': identifier, 'birthrate':birthrate, 'deathrate':deathrate, 'mode': mode, 'nsub': nsubclone, 'scfreq1': frequencies1, 'scfreq2': frequencies2, 'sctime1':time1, 'sctime2':time2})
		dt = pd.concat([dt, add])
	dt.to_csv(out + nm, index = False)

if __name__ == '__main__':
	target, directory, out = vars(arg_parse()).values()
	if out[-1] != '/': out = out + '/'

	if target == 'D1':
		grab_synthetic_E_labels(dir = directory, out = out, nm = 'synthetic_D_labels.csv')

	if target == 'E0':
		metric_simulations(dir = directory, out = out)

	if target == 'E1':
		grab_synthetic_E_labels(dir = directory, out = out, nm = 'synthetic_E_labels.csv')

	if target == 'E2':
		parse_synthetic_E_metrics(dir = directory, out = out, nm = 'synthetic_E_metrics.csv')
	
	if target == 'E3':
		grab_synthetic_E_metric_labels(dir = directory, out = out, nm = 'synthetic_E_labels_metrics.csv')

	if target == 'F1':
		grab_synthetic_F_labels(dir = directory, out = out, nm = 'synthetic_F_labels.csv')