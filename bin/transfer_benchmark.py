#!/bin/env python3
import numpy as np
import pandas as pd
import argparse
import os
import sys
import torch
from nets import *
from utils import *

"""
Run top transfer learning models across TEMULATOR testing set (built with different random seeds)
"""

def arg_parse():
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--dir', type=str, help='Path to directory containing models')
	parser.add_argument('-m', '--model', type=str, help='Model file name')
	parser.add_argument('-dt', '--data', type=str, help='Path to directory containing simulated data')
	parser.add_argument('-od', '--outputdir', type=str, help='Output directory path to save files')
	args = parser.parse_args()
	return args

def load_transfer_model(dir, mod, model_dir = '../analysis/models/'):
	"""
	:param model_dir (str): directory containing pre-trained models 
	:param model_dir (str): directory containing pre-trained models 
	"""
	# Load pre-trained models
	model_dir = os.path.expanduser('../analysis/models/')
	model1="evolution_11_5_Linear_17_9_7_9.027991854570908e-06_5_0.14.TASYG7N3IJR1DLN.pt"
	model2="onesubclone_15_11_Linear_15_3_7_0.0001698628401328024_6.GS3BEXB3O906DHE.pt"
	model1 = load_model(directory = model_dir, mod = model1, input_dim = 192, kind = 'evolution')
	model2 = load_model(directory = model_dir, mod = model2, input_dim = 192, kind = 'onesubclone')
	model1.eval()
	model2.eval()
	pretrained_models = [model1, model2]
	# Initialize model dictionary for transfer learning mode	
	n_linear = int(mod.split('_')[2])
	m = TransferModel(pretrained_models, n_linear = n_linear, gradients = True, input_dim = 192, n_tasks = 4)
	m.load_state_dict(torch.load(dir + mod))
	m.eval()
	return m

def transfer_predict(features, mod, nmc=50):
	mod.train()
	mod.eval()
	"""Takes in numpy array of features and returns predictions for mutrate, birthtime, fitness, frequency"""
	mutrate, birthtime, fitness, frequency = torch.zeros(len(features),1), torch.zeros(len(features),1), torch.zeros(len(features),1), torch.zeros(len(features),1)
	for i in range(nmc):
		mut, birth, fit, freq = mod(torch.Tensor(features))
		mutrate += mut
		birthtime += birth
		fitness += fit
		frequency += freq
		mut, birth, fit, freq = 0, 0, 0, 0
	# Take mean of estimates
	mutrate /= nmc
	birthtime /= nmc
	fitness /= nmc
	frequency /= nmc
	mutrate, birthtime, fitness, frequency = mutrate.detach().numpy().flatten(), birthtime.detach().numpy().flatten(), fitness.detach().numpy().flatten(), frequency.detach().numpy().flatten()
	return (mutrate, birthtime, fitness, frequency)

def prediction_metrics(pred):
	tasks = ['mutrate', 'birthtime', 'fitness', 'frequency']
	mpes, cors = [], []
	for i in tasks:
		truth, prediction = 't_' + i, 'p_' + i
		mpes.append(100 * np.mean( (pred[truth]-pred[prediction]) / pred[truth] ))
		cors.append(np.corrcoef(pred[truth], pred[prediction])[0][1])
	return pd.DataFrame({'mpe': mpes, 'cor':cors, 'task':tasks})

if __name__ == '__main__':
	dir, mod, datadir, outputdir = vars(arg_parse()).values()

	# Load trained transfer learning model
	m = load_transfer_model(dir=dir, mod=mod)

	# Values for rescaling to proper ranges based on max values set or observed in simulated data
	rescale = [500, 8192, 12.25, 1] # Mutrate, time, fitness, frequency	

	# Iterate through all testing data
	preds, metrics = pd.DataFrame(), pd.DataFrame()
	for d in os.listdir(datadir):
		# Load data
		data = np.load(datadir + d, allow_pickle=True)
		seed = np.array([i[0] for i in data])
		depth = np.array([i[1] for i in data])
		features = np.array([i[2][0] for i in data])
		labels = np.array([i[2][1] for i in data])
		
		# Make predictions
		mutrate, birthtime, fitness, frequency = transfer_predict(features=features, mod=m, nmc=25)	
		
		# Build prediction dataframe
		pred = pd.DataFrame({
			'seed':seed,
			'depth':depth,
			't_mutrate':labels[:,0] * rescale[0],
			't_birthtime':np.log2(labels[:,1] * rescale[1]),
			't_fitness':labels[:,2] * rescale[2],
			't_frequency':labels[:,3] * rescale[3],
			'p_mutrate':mutrate * rescale[0],
			'p_birthtime':np.log2(birthtime * rescale[1]),
			'p_fitness':fitness * rescale[2],
			'p_frequency':frequency * rescale[3],
			})
		metric = prediction_metrics(pred)	
		
		# Annotate and concatenate data
		pred['model'], pred['dataset'] = mod, d
		metric['model'], metric['dataset'] = mod, d
		preds, metrics = pd.concat([preds, pred]), pd.concat([metrics, metric])
		pred, metric = 0, 0

	preds, metrics = preds.reset_index(drop=True), metrics.reset_index(drop=True)
	preds.to_csv(outputdir + 'predictions_' + mod + '.csv', index=False)
	metrics.to_csv(outputdir + 'metrics_' + mod + '.csv', index=False)