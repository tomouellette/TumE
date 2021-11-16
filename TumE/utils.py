import numpy as np
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch

"""
General utilities for post-processing of simulations for deep learning based inference and prediction
"""

class OneTargetDataset(Dataset):
  def __init__(self, features, labels):
        self.features = features
        self.one = labels
  def __len__(self):
        return len(self.features)
  def __getitem__(self, index):
        X = self.features[index]
        y1 = self.one[index]
        return X, y1

class TwoTargetDataset(Dataset):
  def __init__(self, features, labels):
        self.features = features
        self.one = labels[0]
        self.two = labels[1]
  def __len__(self):
        return len(self.features)
  def __getitem__(self, index):
        X = self.features[index]
        y1 = self.one[index]
        y2 = self.two[index]                
        return X, y1, y2

class ThreeTargetDataset(Dataset):
  def __init__(self, features, labels):
        self.features = features
        self.one = labels[0]
        self.two = labels[1]
        self.three = labels[2]
  def __len__(self):
        return len(self.features)
  def __getitem__(self, index):
        X = self.features[index]
        y1 = self.one[index]
        y2 = self.two[index] 
        y3 = self.three[index] 
        return X, y1, y2, y3

class FourTargetDataset(Dataset):
  def __init__(self, features, labels):
        self.features = features
        self.one = labels[0]
        self.two = labels[1]
        self.three = labels[2]
        self.four = labels[3]
  def __len__(self):
        return len(self.features)
  def __getitem__(self, index):
        # Load data and get label
        X = self.features[index]
        y1 = self.one[index]
        y2 = self.two[index] 
        y3 = self.three[index] 
        y4 = self.four[index] 
        return X, y1, y2, y3, y4

def list2array(vec):
    """
    Convert list to numpy array
    """
    return np.array([np.array(i) for i in vec])

def data_loaders(train, test, batchsize = 5096):
    """
    Generates pytorch data loaders given a train and test Dataset class
    """
    train_dataloader = DataLoader(train, batch_size=batchsize, shuffle=True, drop_last = True)
    test_dataloader = DataLoader(test, batch_size=batchsize, shuffle=True, drop_last = True)  
    return train_dataloader, test_dataloader

def sim2evo(data):
    """
    Converts simulation output into feature and labels for subclonal parameter inference
    """
    # Load features and targets
    features = np.array([np.hstack(i[0]) for i in data])
    targets = np.array([i[1] for i in data], dtype=object)
    
    # Build arrays for simulations with no subclone
    featuresNS, targetsNS = features[np.where(targets[:,1] == 0)], targets[np.where(targets[:,1] == 0)]
    frequencyNS, timesNS, absfitNS, relfitNS = np.zeros(len(featuresNS)), np.zeros(len(featuresNS)), np.zeros(len(featuresNS)), np.zeros(len(featuresNS))
    modeNS, nsubclonesNS = list2array(targetsNS[:,0]), list2array(targetsNS[:,1])
    
    # Build arrays for simulations with one subclones
    features1, targets1 = features[np.where(targets[:,1] == 1)], targets[np.where(targets[:,1] == 1)]
    frequency1, times1, absfit1, relfit1 = list2array(targets1[:,2]).flatten(), list2array(targets1[:,3]).flatten(), list2array(targets1[:,4]).flatten(), list2array(targets1[:,5]).flatten()
    frequency0, times0, absfit0, relfit0 = np.zeros(len(frequency1)), np.zeros(len(frequency1)), np.zeros(len(frequency1)), np.zeros(len(frequency1))
    mode1, nsubclones1 = list2array(targets1[:,0]), list2array(targets1[:,1])
    
    # Build arrays for simulations with two subclones
    features2, targets2 = features[np.where(targets[:,1] == 2)], targets[np.where(targets[:,1] == 2)]
    frequency2, times2, absfit2, relfit2 = list2array(targets2[:,2]), list2array(targets2[:,3]), list2array(targets2[:,4]), list2array(targets2[:,5])
    frequency2_0, frequency2_1 = frequency2[:,0], frequency2[:,1]
    times2_0, times2_1 = times2[:,0], times2[:,1]
    absfit2_0, absfit2_1 = absfit2[:,0], absfit2[:,1]
    mode2, nsubclones2 = list2array(targets2[:,0]), list2array(targets2[:,1])
    
    # Combine all arrays
    mode, nsubclone = np.hstack((modeNS, mode1, mode2)), np.hstack((nsubclonesNS, nsubclones1, nsubclones2))
    features = np.vstack((featuresNS, features1, features2))
    frequencies1 = np.hstack((frequencyNS, frequency0, frequency2_0))
    frequencies2 = np.hstack((frequencyNS, frequency1, frequency2_1))
    time1 = np.hstack((timesNS, times0, times2_0))
    time2 = np.hstack((timesNS, times1, times2_1))
    fitness1 = np.hstack((absfitNS, absfit0, absfit2_0))
    fitness2 = np.hstack((absfitNS, absfit1, absfit2_1))

    return np.array(features), np.array(mode), np.array(nsubclone), np.array(frequencies1), np.array(frequencies2), np.array(time1), np.array(time2)

def rescale_frequencies(frequencies, lower = 0.09, upper = 0.41, direction = 'train'):
    """
    Rescales frequencies to between 0 and 1 based on lower and upper bound specified in simulations
    """
    if direction == 'train':
        frequencies = (frequencies - lower) / (upper - lower)
    if direction == 'predict':
        frequencies = frequencies * (upper - lower) + lower
    return frequencies

def frequencyThresholds(depth, alt_reads = 2):
    """
    Returns the upper and lower VAF cutoffs based on 2 or 3 binomial standard deviations
    """
    f_min = (alt_reads/depth) + ( ( 2.0*np.sqrt(alt_reads*(1-(alt_reads/depth))) ) / depth)
    f_max = 0.5 - ( ( 3.0*np.sqrt((0.5*depth)*(1-0.5)) ) / depth)
    return f_min, f_max

def uniformdensity(vaf, range_ = [0.02, 0.5], k = 100, depth = 50, alt_reads = 2, cutoff = 1):
    """
    Converts a VAF distribution into a histogram feature vector for training
    """  
    if cutoff == 1:
        f_min, f_max = frequencyThresholds(depth, alt_reads = alt_reads)
        h = np.histogram(vaf[((range_[1] > vaf) & (vaf > f_min))], range = range_, bins = k)
        nd = h[0]
    else:
        h = np.histogram(vaf[((range_[1] > vaf) & (vaf > 0.02))], range = range_, bins = k)
        nd = h[0]
    return nd

def vaf2feature(vaf, depth, k = [64, 128], alt_reads = 2):
    depth = int(np.round(np.mean(depth)))
    features = []
    for bin_number in k:
        ud1 = np.array(uniformdensity(vaf, k = bin_number, depth = depth, alt_reads = alt_reads, range_ = [0.02, 0.5]))
        features.append(ud1)
    return features

def beta_binomial(n, p, rho, size=None):
    mu = p * n
    alpha = (mu / n) * ((1 / rho) - 1)
    beta = n * alpha / mu - alpha
    if beta < 0:
        return 0
    return np.random.binomial(n, np.random.beta(alpha, beta))

def pseudo_vaf(seed = 123, sc = 0, nmuts = 1000, dp = 120, alt_reads = 2, sc_muts = False, with_freq = False, downsample = False):
    np.random.seed(seed)
    f_min = (alt_reads/dp) + ( ( 2.0*np.sqrt(alt_reads*(1-(alt_reads/dp))) ) / dp)
    limit_of_detection = alt_reads / dp
    # Add tail
    # ////////
    from scipy.stats import pareto
    vaf = pareto.rvs(1.4, loc=0.0, scale=f_min, size = nmuts*2)
    vaf = vaf[vaf > f_min]
    depth = np.random.binomial(1e9, p=dp/1e9, size = len(vaf))
    reads = np.array([beta_binomial(i, j, rho = 0.003) for i,j in zip(depth, vaf)])    
    tail = reads / depth    
    # Add subclonal distribution
    # //////////////////////////
    if with_freq == False:
        scfreq = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5][sc]
    else:
        scfreq = sc
    # Add some noise to subclone mean given this is a toy example not a true stochastic simulation where additional muts may arise over time
    if sc_muts == False:
        vaf = np.array([np.random.normal(scfreq, 0.05) for i in range(nmuts)]) 
    else:
        vaf = np.array([np.random.normal(scfreq, 0.05) for i in range(sc_muts)]) 
    vaf = vaf[vaf > f_min]
    depth = np.random.binomial(1e9, p=dp/1e9, size = len(vaf))
    reads = np.array([beta_binomial(i, j, rho = 0.005) for i,j in zip(depth, vaf)])    
    subclone = reads / depth
    subclone = subclone[subclone > 0]
    # Add a clonal peak
    # /////////////////
    vaf = np.array([0.5 for i in range(nmuts)])
    depth = np.random.binomial(1e9, p=dp/1e9, size = nmuts)
    reads = np.array([beta_binomial(i, j, rho = 0.005) for i,j in zip(depth, vaf)])    
    clone = reads / depth
    if downsample != False:
        if downsample > nmuts: 
            raise ValueError('downsample cannot be greater than nmuts')
        else:
            tail = np.random.choice(tail, downsample, replace = False)
    vaf = np.hstack((tail, subclone, clone))
    # Cutoff at LOD
    # /////////////
    vaf = vaf[vaf > limit_of_detection]
    subclone = subclone[subclone > limit_of_detection]
    tail = tail[tail > limit_of_detection]    
    return vaf, subclone, tail, scfreq

def get_bins(subclone, vaf):
    h = np.histogram(vaf, bins = 100)
    gts = lambda x: x[(x > min(subclone)) & (x < max(subclone))]
    locs = [int(np.where(i == h[1])[0]) for i in gts(h[1])]
    return len(locs)