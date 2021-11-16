#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import os
import torch
from functools import partial
import string
import random
from nets import *
from utils import *

"""
Training deep learning models using random hyperparameter search
"""

def arg_parse():
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', '--path', type=str, help='Path to directory containing features and labels from evolutionary simulations (CanEvolve)')
	parser.add_argument('-t', '--task', type=str, help='The task that is being inferred (evolution, onesubclone, twosubclone)')
	parser.add_argument('-od', '--outputdir', type=str, help='Output directory path to save files')
	args = parser.parse_args()
	return args

def random_hyperparameters():
	"""
	Generates a random sample of hyperparameters for training multi-task, multi-convolution networks
	"""
	n_out = int(np.random.choice([i for i in range(4, 32)]))
	n_conv = int(np.random.choice([i for i in range(2, 16)]))
	conv_width = [np.random.choice([1, 3, 5, 7, 9, 11, 13, 15, 17]), np.random.choice([1, 3, 5, 7, 9, 11, 13, 15, 17]), np.random.choice([1, 3, 5, 7, 9, 11, 13, 15, 17])]
	lr = float(10**-np.random.default_rng().uniform(3, 7))	
	branch_type = np.random.choice(['Linear', 'Linear'])
	patience = np.random.choice([3,4,5])
	return n_out, n_conv, branch_type, conv_width, lr, patience

def train_evolution(n_out: int, n_conv: int, branch_type: str, conv_width: list, lr: float, patience: int, directory: str, posweight: float, input_dim = 192, drop = 0.5, epochs = 4) -> tuple:
	"""
	Training function for building models to predict evolutionary mode and the number of subclones
	"""
	model = Evolution(input_dim = input_dim, n_out = n_out, n_conv = n_conv, branch_type = branch_type, conv_width = conv_width, drop = drop)
	net = EvoNet(model, task_types = ['b', 'm'], posweight = posweight)
	mod, dt = net.iterativefit(directory, evolution_loader, batchsize = 256, epochs = epochs, optimizer = 'Adam', learning_rate = lr, patience = patience)
	return (mod, dt)

def train_onesubclone(n_out: int, n_conv: int, branch_type: str, conv_width: list, lr: float, patience: int, directory: str, input_dim = 192, drop = 0.5, epochs = 4) -> tuple:
	"""
	Training function for building models to predict the subclone frequency and subclone timing in the one subclone case
	"""
	model = OneSubclone(input_dim = input_dim, n_out = n_out, n_conv = n_conv, branch_type = branch_type, conv_width = conv_width, drop = drop)
	net = EvoNet(model, task_types = ['r', 'r'])
	mod, dt = net.iterativefit(directory, onesubclone_loader, batchsize = 256, epochs = epochs, optimizer = 'Adam', learning_rate = lr, patience = patience)
	return (mod, dt)

def train_twosubclone(n_out: int, n_conv: int, branch_type: str, conv_width: list, lr: float, patience: int, directory: str, input_dim = 192, drop = 0.5, epochs = 4) -> tuple:
	"""
	Training function for building models to predict the subclone frequencies and subclone timings in the two subclone case
	"""
	model = TwoSubclone(input_dim = input_dim, n_out = n_out, n_conv = n_conv, branch_type = branch_type, conv_width = conv_width, drop = drop)
	net = EvoNet(model, task_types = ['r', 'r', 'r', 'r'])
	mod, dt = net.iterativefit(directory, twosubclone_loader, batchsize = 256, epochs = epochs, optimizer = 'Adam', learning_rate = lr, patience = patience)
	return (mod, dt)

if __name__ == '__main__':
	
	# Parse command-line arguments
	path, task, outputdir = vars(arg_parse()).values()

	# Sample hyperparameters	
	n_out, n_conv, branch_type, conv_width, lr, patience = random_hyperparameters()
	posweight = np.round(np.random.uniform(0.1, 0.5),2) # Only for binary classification weighting

	torch.manual_seed(123456)
	# Train models
	if task == "evolution":
		model, performance = train_evolution(n_out = n_out, n_conv = n_conv, branch_type = branch_type, conv_width = conv_width, lr = lr, patience = patience, directory = path, posweight = posweight, input_dim = 192, drop = 0.5, epochs = 4)
	if task == "onesubclone":
		model, performance = train_onesubclone(n_out = n_out, n_conv = n_conv, branch_type = branch_type, conv_width = conv_width, lr = lr, patience = patience, directory = path, input_dim = 192, drop = 0.5, epochs = 4)
	if task == "twosubclone":
		model, performance = train_twosubclone(n_out = n_out, n_conv = n_conv, branch_type = branch_type, conv_width = conv_width, lr = lr, patience = patience, directory = path, input_dim = 192, drop = 0.5, epochs = 4)
	
	# Annotate files
	params = "_".join([str(n_out), str(n_conv), str(branch_type), str(conv_width[0]), str(conv_width[1]), str(conv_width[2]), str(lr), str(patience), str(posweight)])	
	model_id = ''.join([random.choice(string.ascii_uppercase + string.digits) for i in range(15)])
	
	# Save model and performance metrics
	performance.to_csv(outputdir + task + '_' + params + '.' + model_id + '.csv')	
	torch.save(model.state_dict(), outputdir + task + '_' + params + '.' + model_id + '.pt')