#!/bin/env python3
import numpy as np
import pandas as pd
import argparse
import os
import torch
from nets import *
from utils import *

"""
Run top models across simulated/synthetic datasets
"""

def arg_parse():
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--dir', type=str, help='Path to directory containing models')
	parser.add_argument('-m', '--model', type=str, help='Model file name')
	parser.add_argument('-dt', '--data', type=str, help='Path to directory containing simulated data')
	parser.add_argument('-od', '--outputdir', type=str, help='Output directory path to save files')
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	modeldir, mod, datadir, outputdir = vars(arg_parse()).values()
	modtype = mod.split('_')[0]
	dt = pd.DataFrame()
	for i in [j for j in os.listdir(datadir) if '.npy' in j]:
		path = datadir + i
		
		if modtype == 'evolution':
			mode, nsub = prediction(directory = modeldir, mod = mod, path = datadir + i, kind = modtype, means = False)
			mode_prob_mean = np.mean(mode, axis = 0) # Mean probability estimates for P(Selection)
			mode_prob_q89 = np.quantile(mode, q = 0.055, axis = 0) # Probability estimates by controlling for variance in approximate posterior (lower 89% bound)
			cmode_mean = np.array([int(i > 0.5) for i in mode_prob_mean]) # Selection classification using mean
			cmode_q89 = np.array([int(i > 0.5) for i in mode_prob_q89]) # Selection classification using q89 lower bound
			cnsub = np.argmax(np.mean(nsub, axis=0), axis = 1) # Number of subclone predictions
			cnsub_q89 = np.array([0 if mode_prob_q89[i] < 0.5 else cnsub[i] for i in range(len(cnsub))]) # Number of subclones predictions conditioned on q89 selection
			identifier = [i.split('_')[0] + '_' + str(m) for m in range(len(cmode_mean))]
			add = pd.DataFrame({'identifier':identifier, 'TumE_mode_mean':cmode_mean, 'TumE_mode_mean':cmode_q89, 'TumE_mode_prob_mean':mode_prob_mean, 'TumE_mode_prob_q89':mode_prob_q89, 'TumE_nsub':cnsub, 'Tume_nsub_q89':cnsub_q89})

		if modtype == 'onesubclone':
			freq, time = prediction(directory = modeldir, mod = mod, path = datadir + i, kind = modtype)
			identifier = [i.split('_')[0] + '_' + str(m) for m in range(len(freq))]
			add = pd.DataFrame({'identifier':identifier, 'TumE_f':freq, 'TumE_t':time})
		
		if modtype == 'twosubclone':
			freq1, freq2, time1, time2 = prediction(directory = modeldir, mod = mod, path = datadir + i, kind = modtype)
			identifier = [i.split('_')[0] + '_' + str(m) for m in range(len(freq1))]
			add = pd.DataFrame({'identifier':identifier, 'TumE_f1':freq1, 'TumE_f2':freq2, 'TumE_t1':time1, 'TumE_t2':time2})
		
		dt = pd.concat([dt, add])

	dt.to_csv(outputdir + 'predictions_' + mod + '.csv')