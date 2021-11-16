#!/bin/env python3
import os
import numpy as np
from transfer import run_temulator, Sampler
from temulator import Temulator
import pandas as pd
import argparse
import random
import string

"""
Build a random sample of training data for evaluation of trained transfer learning models
"""

def arg_parse():
	parser = argparse.ArgumentParser()
	parser.add_argument('-v', '--viable', type=str, help='Path to tab-separated file containing viable (detectable subclone) birthtime-birthrate combinations')
	parser.add_argument('-n', '--nsims', type=int, help='The number of simulations to generate')
	parser.add_argument('-o', '--output', type=str, help='The output directory to save simulations')
	parser.add_argument('-s', '--seed', type=int, help='Seed to use for this set of simulations')
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	# Parse arguments
	viable, nsims, out, seed = vars(arg_parse()).values()

	# Hard code paths > need custom R install to ensure libR.so is available on cluster for rpy2
	os.environ['R_HOME'] = '/.mounts/labs/awadallalab/private/touellette/sources/R-4.0.3/'

	# Load viable
	viable = pd.read_csv(viable, delimiter='\t')
	viable = viable[viable['b'] > 1] # Only keep birthrates that are positively selected
	viable = viable.groupby(['b', 't']).mean().reset_index()

	# Ensure seeds do not overlap with testing data (keep below 1e5)
	train_set = []
	maximum_label_values = [500, 2**np.max(viable['t']), (np.max(viable['b']) - 0.2)/(1-0.2), 1] # Normalize labels to ~0-1 based on maxed simulated values
	sampler = Sampler(viable)
	for i in range(nsims):
		parameters = sampler.get_parameters()
		sim = run_temulator(parameters, maximum_label_values = maximum_label_values, seed = seed)
		train_set.append(np.array([seed, sim], dtype=object))

	sim_id = ''.join([random.choice(string.ascii_uppercase + string.digits) for i in range(15)])
	train_set = np.array(train_set, dtype=object)	

	np.save(out + sim_id + str(seed) + '.npy' , train_set)