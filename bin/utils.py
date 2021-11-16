import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch

"""
General utilities for processing of simulations for deep learning based inference and prediction
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

    :param data (list/np.ndarray) An array of simulated data from autoSimulation output in CanEvolve.jl where each datapoint is a tuple of (feature, labels))
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
    times2_0, times2_1 = np.transpose(np.take_along_axis(np.transpose([times2_0, times2_1]), np.argsort(np.transpose([frequency2_0, frequency2_1]), axis = 1), axis = 1))
    absfit2_0, absfit2_1 = np.transpose(np.take_along_axis(np.transpose([absfit2_0, absfit2_1]), np.argsort(np.transpose([frequency2_0, frequency2_1]), axis = 1), axis = 1))
    frequency2_0, frequency2_1 = np.transpose(np.take_along_axis(np.transpose([frequency2_0, frequency2_1]), np.argsort(np.transpose([frequency2_0, frequency2_1]), axis = 1), axis = 1))
    
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

    :param frequencies (np.ndarray): Rescales frequencies from simulated range to [0,1] or training range back to [lower,upper]
    :param lower (float): Lower bound of simulated ranges for subclone frequencies
    :param upper (float): Upper bound of simulated ranges for subclone frequencies
    :param direction (str): Direction for converting ranges

    >return (np.ndarray) An array of rescaled frequencies
    """
    if direction == 'train':
        frequencies = (frequencies - lower) / (upper - lower)
    if direction == 'predict':
        frequencies = frequencies * (upper - lower) + lower
    return frequencies

def evolution_loader(directory, file, batchsize = 256, sample_size = 100, bias = 2, testset = False):
    """
    Returns a dataloader for subclone frequency inference for N subclones (1 or 2) given a simulation file

    :param directory (str): The directory containing CanEvolve simulated data
    :param file (str): The specific file to be loaded for training
    :param batchsize (int): The size of each training batch
    :param sample_size (int): The size of groups to upsample/resample positive selection simulations. Note that test data will be re-balanced too. 
    :param bias (int): The number of one subclone samples relative to two subclone samples. Higher values put less weight on two subclone estimates to control overconfidence in less parsimonious settings.
    :testset (bool): Indicates if this file is used for testing. If True, will return features for each test sample in numpy format along with a dataloader

    >return (torch.utils.data.DataLoader) A PyTorch DataLoader
    """
    data = np.vstack(np.load(directory + file, allow_pickle=True))
    features, mode, nsubclone, frequencies1, frequencies2, _, _ = sim2evo(data)
    data = 0        
    # Force balancing across subclone frequencies
    features_0, mode_0, nsubclone_0 = features[np.where(nsubclone == 0)], mode[np.where(nsubclone == 0)], nsubclone[np.where(nsubclone == 0)]
    features_1, mode_1, nsubclone_1 = features[np.where(nsubclone == 1)], mode[np.where(nsubclone == 1)], nsubclone[np.where(nsubclone == 1)]
    features_2, mode_2, nsubclone_2 = features[np.where(nsubclone == 2)], mode[np.where(nsubclone == 2)], nsubclone[np.where(nsubclone == 2)]    
    
    # Balance 1 subclone simulations with respect to subclone frequency (up to 2 decimal places)
    f1 = frequencies2[np.where(nsubclone == 1)].round(2)        
    f1 = np.hstack([np.random.choice(np.where(f1 == i)[0], size = int(len(features_0)/len(np.unique(f1))), replace = True) for i in np.unique(f1)])     
    features_1, mode_1, nsubclone_1= features_1[f1], mode_1[f1], nsubclone_1[f1]        
    
    # Balance 2 subclone simulations with respect to frequency and distance between subclones
    frequencies2_1, frequencies2_2 = frequencies1[np.where(nsubclone == 2)], frequencies2[np.where(nsubclone == 2)]
    f2 = np.where((frequencies2_2 - frequencies2_1) > 0.05)
    frequencies2_1, frequencies2_2 = frequencies2_1[f2], frequencies2_2[f2]
    f2 = (frequencies2_2-frequencies2_1).round(2)    
    f2 = np.hstack([np.random.choice(np.where(f2 == i)[0], size = int(len(features_0)/len(np.unique(f2))), replace = True) for i in np.unique(f2)]) 
    features_2, mode_2, nsubclone_2 = features_2[f2], mode_2[f2], nsubclone_2[f2]    
    
    # Recombine features and class labels
    features = np.vstack((features_0, features_1, features_2))
    mode = np.hstack((mode_0, mode_1, mode_2))
    nsubclone = np.hstack((nsubclone_0, nsubclone_1, nsubclone_2))
    features_0, mode_0, nsubclone_0, features_1, mode_1, nsubclone_1, features_2, mode_2, nsubclone_2, = 0, 0, 0, 0, 0, 0, 0, 0, 0

    # Dataloader
    dataset = TwoTargetDataset(np.array(features), (mode, nsubclone))
    _, mode, nsubclone, _, _, _, _, _, _ = 0, 0, 0, 0, 0, 0, 0, 0, 0
    dataload = DataLoader(dataset, batch_size = batchsize, shuffle=True, drop_last = True)    
    if testset == True:
        return dataload, features        
    else:
        return dataload

def onesubclone_loader(directory, file, batchsize = 256, sample_size = 350, testset = False):
    """
    Returns a dataloader for subclone frequency inference for N subclones (1 or 2) given a simulation file

    >> See evolution_loader for function description
    """
    data = np.vstack(np.load(directory + file, allow_pickle=True))
    features, _, nsubclone, _, frequencies2, _, times2 = sim2evo(data)
    features, frequencies2, times2 = features[np.where(nsubclone == 1)], frequencies2[np.where(nsubclone == 1)], times2[np.where(nsubclone == 1)]
    
    # Balance frequencies up to 2 decimal places
    f1 = np.hstack([np.random.choice(np.where(frequencies2.round(2) == i)[0], size = sample_size, replace = True) for i in np.unique(frequencies2.round(2))])
    features, frequencies2, times2 = features[f1], frequencies2[f1], times2[f1]
    
    # Rescale frequencies between 0 and 1
    frequencies2 = rescale_frequencies(frequencies2, direction = 'train')
    data = 0
    
    # Dataloader    
    dataset = TwoTargetDataset(np.array(features), (frequencies2, times2))
    _, _, nsubclone, _, frequencies2, _, times2 = 0, 0, 0, 0, 0, 0, 0
    dataload = DataLoader(dataset, batch_size = batchsize, shuffle=True, drop_last = True)    
    if testset == True:
        return dataload, features        
    else:
        return dataload

def twosubclone_loader(directory, file, batchsize = 256, sample_size = 350, testset = False):
    """
    Returns a dataloader for subclone frequency inference for N subclones (1 or 2) given a simulation file

    >> See evolution_loader for function description
    """
    data = np.vstack(np.load(directory + file, allow_pickle=True))
    features, _, nsubclone, frequencies1, frequencies2, times1, times2 = sim2evo(data)
    features, frequencies1, frequencies2, times1, times2 = features[np.where(nsubclone == 2)], frequencies1[np.where(nsubclone == 2)], frequencies2[np.where(nsubclone == 2)], times1[np.where(nsubclone == 2)], times2[np.where(nsubclone == 2)]
    data = 0
    
    # Balance 2 subclone simulations with respect to frequency and distance between subclones
    f2 = np.where((frequencies2 - frequencies1) > 0.05)
    features, frequencies1, frequencies2, times1, times2 = features[f2], frequencies1[f2], frequencies2[f2], times1[f2], times2[f2]
    f2 = (frequencies2-frequencies1).round(2)
    f2 = np.hstack([np.random.choice(np.where(f2 == i)[0], size = sample_size, replace = True) for i in np.unique(f2)]) 
    features, frequencies1, frequencies2, times1, times2 = features[f2], frequencies1[f2], frequencies2[f2], times1[f2], times2[f2]

    # Rescale frequencies between 0 and 1
    frequencies1 = rescale_frequencies(frequencies1, direction = 'train')
    frequencies2 = rescale_frequencies(frequencies2, direction = 'train')

    # Dataloader
    dataset = FourTargetDataset(np.array(features), (frequencies1, frequencies2, times1, times2))
    _, _, nsubclone, frequencies1, frequencies2, times1, times2 = 0, 0, 0, 0, 0, 0, 0
    dataload = DataLoader(dataset, batch_size = batchsize, shuffle=True, drop_last = True)
    if testset == True:
        return dataload, features        
    else:
        return dataload

def temulator_loader(directory, file, batchsize = 256, testset = False):
    """
    Returns a dataloader for VAF distributions generated by TEMULATOR with labels of mutation rate, birth times, fitness, clone frequencies
    """
    data = np.vstack(np.load(directory + file, allow_pickle=True))
    features = np.array([i[1][0] for i in data])
    labels = np.array([i[1][1] for i in data])
    
    # Dataloader    
    dataset = FourTargetDataset(np.array(features), (labels[:,0], labels[:,1], labels[:,2], labels[:,3]))    
    dataload = DataLoader(dataset, batch_size = batchsize, shuffle=True, drop_last = True)            
    if testset == True:
        return dataload, features        
    else:
        return dataload

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