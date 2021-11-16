import os
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
from utils import vaf2feature
import pandas as pd
import multiprocessing
from functools import partial
import argparse

N_PROCESSORS = 20

"""
Checking the specification of training data set with respect to patients' sequenced tumour biopsies at the feature level
"""

def arg_parse():
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', '--path', type=str, help='Path to directory containing features and labels from evolutionary simulations (CanEvolve)')
	parser.add_argument('-e', '--emp', type=str, help='Path to file containing features for empirical patient samples')
	parser.add_argument('-od', '--outputdir', type=str, help='Output directory path to save files')
	args = parser.parse_args()
	return args

def get_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
	"""
	Note that correlation distance is actually 1 - correlation
	"""
	return np.array([distance.euclidean(x, y), distance.cityblock(x, y)])

def get_nearest(x: np.ndarray, synthetic: str, top = 2) -> tuple:
	"""
	Computes the top N nearest neighbours between x and a set of synthetic features

	:param x (np.ndarray) Feature vector for empirical distribution
	:param synthetic (str) Path to simulated tumour numpy array
	:param top (int) The number of closest nearest neighbours to subset for each distance metric
	"""	
	# Load  features and labels
	synthetic = np.load(synthetic, allow_pickle=True)
	features, labels = np.array([np.hstack(i[0]) for i in synthetic]), np.array([i[1] for i in synthetic])	
	# Compute distances
	e, c = np.transpose(np.array([get_distances(x, i) for i in features]))	
	# Get indices for nearest neighbours based on each distance metric
	ei, ci = np.argsort(e)[0:top], np.argsort(c)[0:top]
	indices = np.union1d(ei, ci)
	# Return nearest neigbours
	nearest_neighbours = (e[indices], c[indices], features[indices], labels[indices])
	return nearest_neighbours

def parallel_neighbours(processed, path):
	empirical_features_meanDP, empirical_features_meanEC, sample_id = processed
	nnDP, nnEC = [], []
	for j in os.listdir(path):
		nnd = get_nearest(x = empirical_features_meanDP, synthetic = path + j)
		nne = get_nearest(x = empirical_features_meanEC, synthetic = path + j)
		nnDP.append(nnd), nnEC.append(nne)
	return np.array([sample_id, nnDP, nnEC], dtype=object)

if __name__ == '__main__':
	
	# Parse command-line arguments
	path, empirical, outputdir = vars(arg_parse()).values()
	
	# Generate features for empirical data # sample_id, VAF_adjust, meanDP, meanEC
	empirical = pd.read_csv(empirical)
	empirical_features_meanDP, empirical_features_meanEC, sample_ids = [], [], []
	counter = 0
	for sample_id in np.unique(empirical['sample_id']):
		counter += 1
		print(counter)
		sample = empirical[empirical['sample_id'] == sample_id]
		addDP = np.hstack(vaf2feature(np.array(sample['VAF_adjust']), np.array(sample['meanEC'])[0], k = [64, 128]))
		addEC = np.hstack(vaf2feature(np.array(sample['VAF_adjust']), np.array(sample['meanDP'])[0], k = [64, 128]))
		empirical_features_meanDP.append(addDP)
		empirical_features_meanEC.append(addEC)
		sample_ids.append(sample_id)

	empirical_processed = [(i,j,z) for i,j,z in zip(empirical_features_meanDP, empirical_features_meanEC, sample_ids)]

	# Partial functions to ensure multiprocessing map applies to only the index argument
	partial_neighbours=partial(parallel_neighbours, path = path)

	# Generate vaf density vectors
	pool = multiprocessing.Pool(N_PROCESSORS)
	neighbours = np.array(pool.map(partial_neighbours, empirical_processed), dtype=object)
	pool.close()  
	pool.terminate()
	pool.join() 
	
	np.save(outputdir + 'specification-nearest_neighbours.npy', neighbours)