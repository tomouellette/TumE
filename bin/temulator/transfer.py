import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd 
import numpy as np
from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF    
import copy
from temulator import Temulator
import multiprocessing
from functools import partial
import argparse
import math

"""
Extra
Some test code for on-the-fly transfer learning for applications to alternative simulation/inference schemes
"""
def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='Path to directory containing features and labels from evolutionary simulations (CanEvolve)')
    parser.add_argument('-e', '--emp', type=str, help='Path to file containing features for empirical patient samples')
    parser.add_argument('-od', '--outputdir', type=str, help='Output directory path to save files')
    args = parser.parse_args()
    return args

class FiveTargetDataset(Dataset):
  def __init__(self, features, labels):
        self.features = features
        self.one = labels[0]
        self.two = labels[1]
        self.three = labels[2]
        self.four = labels[3]
        self.five = labels[4]
  def __len__(self):
        return len(self.features)
  def __getitem__(self, index):
        # Load data and get label
        X = self.features[index]
        y1 = self.one[index]
        y2 = self.two[index] 
        y3 = self.three[index] 
        y4 = self.four[index]
        y5 = self.five[index]
        return X, y1, y2, y3, y4, y5

def load_model(mod: str, kind: str, input_dim = 192, external = False):
    """
    Loads a TumE PyTorch model state dictionary

    :param mod (str): The name of pytorch model in standardized format (see line 1 split method for structure)
    :param kind (str): One of evolution, onesubclone, or twosubclone
    :param input_dim (int): The width of feature vector
    :param external (bool): If False, models will be loaded from internal TumE library, else mod references a direct path to a pytorch model
    """
    if kind == 'evolution':
        _, name, n_out, n_conv, branch_type, conv_width1, conv_width2, conv_width3, lr, patience, extra = mod.split('_')
    else:    
        _, name, n_out, n_conv, branch_type, conv_width1, conv_width2, conv_width3, lr, extra = mod.split('_')
    if kind == 'evolution':
        m = Evolution(input_dim = int(input_dim), n_out = int(n_out), n_conv = int(n_conv), branch_type = branch_type, conv_width = [int(conv_width1), int(conv_width2), int(conv_width3)], drop = 0.5)
    if kind == 'onesubclone':
        m = OneSubclone(input_dim = int(input_dim), n_out = int(n_out), n_conv = int(n_conv), branch_type = branch_type, conv_width = [int(conv_width1), int(conv_width2), int(conv_width3)], drop = 0.5)
    if kind == 'twosubclone':
        m = TwoSubclone(input_dim = int(input_dim), n_out = int(n_out), n_conv = int(n_conv), branch_type = branch_type, conv_width = [int(conv_width1), int(conv_width2), int(conv_width3)], drop = 0.5)
    if external == False:
        mod = importlib.resources.open_binary('TumE.models', mod)
    m.load_state_dict(torch.load(mod))
    m.eval()
    return m

class EarlyStopping:
    """
    Basic early stopping function for regularization in time

    :param patience (int): Number of epochs without a change in loss before terminating training
    :param delta (int): The required change in loss to consider an epoch as not improved for early stopping
    :param decimals (int): Number of decimals to allow when computing change in loss between epochs   
    """
    def __init__(self, patience=3, delta = 0, decimals = 8):
        self.patience = patience + 1
        self.losses = []
        self.delta = delta
        self.decimals = decimals
    def update(self, loss):
        if len(self.losses) < self.patience:
          self.losses.append(loss)
        else:
          for i in range(1, self.patience, 1):
            self.losses[i-1] = self.losses[i]
          self.losses[self.patience-1] = round(loss, self.decimals)
    def check(self):
        if len(self.losses) == self.patience:
            i1 = np.array(self.losses)[0:(self.patience-1)]
            i2 = np.array(self.losses)[1:self.patience]
            diff = i1 - i2
            if np.sum(diff <= self.delta) >= (self.patience - 1):
              return 1
            else:
              return 0
        else:
            return 0

def InitializeLoss(task_type: str) -> torch.nn.modules.loss:
    """
    Loss functions for binary classification (b), multi-class classification (m), and regression (r)
    """
    if task_type == 'b':
        return nn.BCEWithLogitsLoss()
    elif task_type == 'm':
        return nn.CrossEntropyLoss()
    elif task_type == 'r':
        return nn.MSELoss()
    else:
        print("Improper specification of task type, please indicate b, m, or r.")

def GetLoss(Ys: torch.Tensor, yhats: torch.Tensor, loss_functions: torch.nn.modules.loss, task_types: str, device) -> torch.Tensor:
    """
    Computing loss for binary classification (b), multi-class classification (m), and regression (r)
    """
    losses = torch.zeros(len(task_types))
    for task_ind in range(len(task_types)):
        if len(task_types) == 1:
            yhat, y, task = yhats.to(device, non_blocking = True), Ys[task_ind].to(device, non_blocking = True), task_types[task_ind]
        else:
            yhat, y, task = yhats[task_ind].to(device, non_blocking = True), Ys[task_ind].to(device, non_blocking = True), task_types[task_ind]
        if task_types[task_ind] in ['b', 'r']:
            if len(task_types[task_ind]) < 2:
                y = y.unsqueeze(1)    
            losses[task_ind] = loss_functions[task_ind](yhat, y.float())
        else:
            losses[task_ind] = loss_functions[task_ind](yhat, y.long())
    cumulative_loss = torch.sum(losses)
    return cumulative_loss

class Sampler:
    """
    Random sampling of parameters in a the viable search space  
    
    :param viable (list): A data frame containing viable birthrate time combinations

    > We explore:
        - Mutation rate: 1 - 500
        - Number clonal: 1 - 5000
        - Sequencing depth: 70 - 350
        - Birth rates and times based on ranges where detectable subclone arises at N=1e6 (see viable_1subclone_parameters.tsv)
    """
    def __init__(self, viable, lower_bound = 0.09, upper_bound = 0.41):
        viable = viable[(viable['f'] > lower_bound) & (viable['f'] < upper_bound)].reset_index()
        self.viable = viable
        #
        # Build noisy sampling distribution for viable birthtime-birthrate combinations
        kernel = RBF(length_scale=100.0) + DotProduct() + WhiteKernel()
        gpr = GaussianProcessRegressor(kernel=kernel,alpha=1e-6, random_state=0).fit(np.array(viable['b']).reshape(len(viable['b']), 1), np.array(viable['t']))
        self.sample_birthtime = lambda x: gpr.sample_y([[x]], random_state=np.random.randint(1e4)).flatten()[0]
        self.sample_birthrate = lambda: np.random.uniform(np.min(viable['b']), np.max(viable['b']))
    #
    def get_parameters(self):
        # Sample non-fixed parameters
        birthtime = 1e5
        while birthtime > 1e4:
            mutrate = np.random.randint(1, 500)
            nclonal = int(np.random.randint(1, 5000))
            birthrate = self.sample_birthrate()
            birthtime = int(2**self.sample_birthtime(birthrate))
            depth = np.random.randint(70, 350)
            # Build parameter dictionary
            parameters = {
            'birthrates': [1, birthrate],
            'deathrates': [0.2, 0.2],
            'mutation_rates': [mutrate, mutrate],
            'clone_start_times': [0, birthtime],
            'fathers': [0, 0],
            'simulation_end_time': 1e4,
            'number_clonal_mutations': nclonal,
            'depth': depth,
            'depth_model': 2,
            'min_vaf': 0.05,
            'purity': 1
            }
        return parameters

def run_temulator(parameters, maximum_label_values, seed = 0): 
    """
    Run temulator using hardcoded parameter ranges in sample_parameters()
    Re-scale all labels to approximately between 0 and 1
    """
    simulator = Temulator(silent = True)
    output = simulator.run(parameters = parameters, seed = seed)
    features, labels = simulator.process()
    labels = [i[0] if type(i) == np.ndarray else i for i in labels]
    # Labels: mutrate, birth_times, fitness, frequencies
    labels = [labels[0]/maximum_label_values[0], 
              labels[1]/maximum_label_values[1], 
              labels[2]/maximum_label_values[2],
              labels[3]/maximum_label_values[3]]
    return np.array([features, np.array(labels)], dtype=object)

class Evolution(nn.Module):
    """
    Convolutional multi-task architecture to infer global evolutionary/genomic features    
    
    :param input_dim (int): Length of 1-dimensional feature vector
    :param n_out (int): Number of output channels/feature maps for each convolutional layer
    :param n_conv (int): Number of convolutional layers in trunks of network
    :param conv_width (list): Kernel size for convolution layers (where 1st index is for trunk 1, 2nd index is for trunk 2, and 3rd index is for branches if branch_type != 'Linear')
    :param drop (float): Dropout probability following each layer (Default to 0.5 for MC dropout)
    :param branch_type (str): Specifies if output branches should pass through fully-connected (Linear) or global average pooling (Default 'Linear')
    """
    def __init__(self, input_dim = 192, n_out = 4, n_conv = 4, conv_width = [3, 9, 17], drop = 0.5, branch_type = 'Linear'):
        super(Evolution, self).__init__() 
        self.n_conv = n_conv
        self.branch_type = branch_type

        # Build the convolutional trunks of network
        convolution1, convolution2 = [], []
        for i in range(1, n_conv + 1):
            if i == 1:
                convolution1.append(nn.Conv1d(in_channels = 1, out_channels = n_out, kernel_size = conv_width[0], padding = math.floor(conv_width[0]/2)))
                convolution2.append(nn.Conv1d(in_channels = 1, out_channels = n_out, kernel_size = conv_width[1], padding = math.floor(conv_width[1]/2)))
            else:
                convolution1.append(nn.Conv1d(in_channels = n_out, out_channels = n_out, kernel_size = conv_width[0], padding = math.floor(conv_width[0]/2)))
                convolution2.append(nn.Conv1d(in_channels = n_out, out_channels = n_out, kernel_size = conv_width[1], padding = math.floor(conv_width[1]/2)))
            convolution1.append(nn.Hardswish())
            convolution2.append(nn.Hardswish())
            convolution1.append(nn.Dropout(drop))
            convolution2.append(nn.Dropout(drop))
        self.convolution1 = nn.Sequential(*convolution1)
        self.convolution2 = nn.Sequential(*convolution2)

        # Build the task-specific branches of the network where 'Linear' builds fully-connected layers and 'GAP' (or any other string) builds global average pooling branches
        if branch_type == 'Linear':
            self.mode = nn.Sequential(            
                nn.Flatten(),
                nn.Linear(input_dim*n_out, 64),            
                nn.Hardswish(),
                nn.Dropout(p = drop),
                nn.Linear(64, 32),
                nn.Hardswish(),
                nn.Dropout(p = drop),
                nn.Linear(32, 1)            
            )
            self.nsubclones = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim*n_out, 64),            
                nn.Hardswish(),
                nn.Dropout(p = drop),
                nn.Linear(64, 32),
                nn.Hardswish(),
                nn.Dropout(p = drop),
                nn.Linear(32, 3)
            )
        else: # Global average pooling
            self.mode = nn.Sequential(
                nn.Conv1d(in_channels = n_out, out_channels = n_out, kernel_size = conv_width[2], padding = math.floor(conv_width[2]/2)),
                nn.Hardswish(),
                nn.Dropout(p = drop),
                nn.Conv1d(in_channels = n_out, out_channels = 1, kernel_size = conv_width[2], padding = math.floor(conv_width[2]/2)),
                nn.Hardswish(),
                nn.Dropout(p = drop),
                nn.AdaptiveAvgPool1d(1)
            )
            # Number of subclones (0, 1, or 2)
            self.nsubclones = nn.Sequential(
                nn.Conv1d(in_channels = n_out, out_channels = n_out, kernel_size = conv_width[2], padding = math.floor(conv_width[2]/2)),
                nn.Hardswish(),
                nn.Dropout(p = drop),
                nn.Conv1d(in_channels = n_out, out_channels = 1, kernel_size = conv_width[2], padding = math.floor(conv_width[2]/2)),
                nn.Hardswish(),
                nn.Dropout(p = drop),
                nn.AdaptiveAvgPool1d(3)
            )                     
    def forward(self, x):
        x1 = self.convolution1(x.unsqueeze(1).float())
        x2 = self.convolution2(x.unsqueeze(1).float())
        t10 = self.mode(x1.float())
        t11 = self.mode(x2.float())
        t20 = self.nsubclones(x1.float())
        t21 = self.nsubclones(x2.float())        
        if self.branch_type == 'Linear':
            t1 = torch.mean(torch.cat((t10, t11), dim = 1), dim=1)
            t2 = torch.mean(torch.cat((t20.unsqueeze(1), t21.unsqueeze(1)), dim = 1), dim=1)
            return t1.unsqueeze(1), t2
        else:
            t1 = torch.mean(torch.cat((t10, t11), dim = 1), dim=1)
            t2 = torch.mean(torch.cat((t20, t21), dim = 1), dim=1)
            return t1, t2

class OneSubclone(nn.Module):
    """
    Convolutional multi-task architecture to parse frequency and timing of a single subclone

    :param input_dim (int): Length of 1-dimensional feature vector
    :param n_out (int): Number of output channels/feature maps for each convolutional layer
    :param n_conv (int): Number of convolutional layers in trunks of network
    :param conv_width (list): Kernel size for convolution layers (where 1st index is for trunk 1, 2nd index is for trunk 2, and 3rd index is for branches if branch_type != 'Linear')
    :param drop (float): Dropout probability following each layer (Default to 0.5 for MC dropout)
    :param branch_type (str): Specifies if output branches should pass through fully-connected (Linear) or global average pooling (Default 'Linear')
    """
    def __init__(self, input_dim = 192, n_out = 4, n_conv = 4, conv_width = [3, 9, 17], drop = 0.5, branch_type = 'Linear'):
        super(OneSubclone, self).__init__() 
        self.n_conv = n_conv
        self.branch_type = branch_type

        # Build the convolutional trunks of network
        convolution1, convolution2 = [], []
        for i in range(1, n_conv + 1):
            if i == 1:
                convolution1.append(nn.Conv1d(in_channels = 1, out_channels = n_out, kernel_size = conv_width[0], padding = math.floor(conv_width[0]/2)))
                convolution2.append(nn.Conv1d(in_channels = 1, out_channels = n_out, kernel_size = conv_width[1], padding = math.floor(conv_width[1]/2)))
            else:
                convolution1.append(nn.Conv1d(in_channels = n_out, out_channels = n_out, kernel_size = conv_width[0], padding = math.floor(conv_width[0]/2)))
                convolution2.append(nn.Conv1d(in_channels = n_out, out_channels = n_out, kernel_size = conv_width[1], padding = math.floor(conv_width[1]/2)))
            convolution1.append(nn.Hardswish())
            convolution2.append(nn.Hardswish())
            convolution1.append(nn.Dropout(drop))
            convolution2.append(nn.Dropout(drop))
        self.convolution1 = nn.Sequential(*convolution1)
        self.convolution2 = nn.Sequential(*convolution2)
        
        # Build the task-specific branches of the network where 'Linear' builds fully-connected layers and 'GAP' (or any other string) builds global average pooling branches
        if branch_type == 'Linear':
            self.frequency1 = nn.Sequential(            
                nn.Flatten(),
                nn.Linear(input_dim*n_out, 64),            
                nn.Hardswish(),
                nn.Dropout(p = drop),
                nn.Linear(64, 32),
                nn.Hardswish(),
                nn.Dropout(p = drop),
                nn.Linear(32, 1),
                nn.ReLU()          
            )
            self.timing1 = nn.Sequential(            
                nn.Flatten(),
                nn.Linear(input_dim*n_out, 64),            
                nn.Hardswish(),
                nn.Dropout(p = drop),
                nn.Linear(64, 32),
                nn.Hardswish(),
                nn.Dropout(p = drop),
                nn.Linear(32, 1),
                nn.ReLU()           
            )
        else: # Global average pooling
            self.frequency1 = nn.Sequential(            
                nn.Conv1d(in_channels = n_out, out_channels = n_out, kernel_size = conv_width[2], padding = math.floor(conv_width[2]/2)),
                nn.Hardswish(),
                nn.Dropout(p = drop),
                nn.Conv1d(in_channels = n_out, out_channels = 1, kernel_size = conv_width[2], padding = math.floor(conv_width[2]/2)),
                nn.Hardswish(),
                nn.Dropout(p = drop),
                nn.AdaptiveAvgPool1d(1),
                nn.ReLU()
            )
            self.timing1 = nn.Sequential(            
                nn.Conv1d(in_channels = n_out, out_channels = n_out, kernel_size = conv_width[2], padding = math.floor(conv_width[2]/2)),
                nn.Hardswish(),
                nn.Dropout(p = drop),
                nn.Conv1d(in_channels = n_out, out_channels = 1, kernel_size = conv_width[2], padding = math.floor(conv_width[2]/2)),
                nn.Hardswish(),
                nn.Dropout(p = drop),
                nn.AdaptiveAvgPool1d(1),
                nn.ReLU() 
            )               
    def forward(self, x):
        x1 = self.convolution1(x.unsqueeze(1).float())
        x2 = self.convolution2(x.unsqueeze(1).float())     
        t10, t11 = self.frequency1(x1.float()), self.frequency1(x2.float())
        t20, t21 = self.timing1(x1.float()), self.timing1(x2.float())
        if self.branch_type == 'Linear':
            t1 = torch.mean(torch.cat((t10, t11), dim = 1), dim=1)
            t2 = torch.mean(torch.cat((t20, t21), dim = 1), dim=1)
            return t1.unsqueeze(1), t2.unsqueeze(1)
        else:
            t1 = torch.mean(torch.cat((t10, t11), dim = 1), dim=1)
            t2 = torch.mean(torch.cat((t20, t21), dim = 1), dim=1)
            return t1, t2
            
class TransferModel(nn.Module):
    """
    Takes pre-trained TumE models and mutates network for transfer learning on TEMULATOR simulations
    
    :param pretrained_models (list): A list of TumE models, takes each model and uses it as a tunable feature extractor
    :input_dim (int): The input feature width
    """
    def __init__(self, pretrained_models, input_dim = 192, n_tasks = 4):
        super(TransferModel, self).__init__()
        # Extract convolution layers from pre-trained TumE model(s)
        convolutions = []
        nchannels = []
        for i in pretrained_models:
            convolutions.append(nn.Sequential(*list(i.children())[0]))
            convolutions.append(nn.Sequential(*list(i.children())[1]))
            nchannels.append(list(i.children())[0][0].weight.size()[0])
            nchannels.append(list(i.children())[1][0].weight.size()[0])            
        self.convolutions = nn.ModuleList(convolutions)
        #
        # Build fully connected layers
        n_channels = np.sum(nchannels)
        fc = []
        for i in range(n_tasks):
            fc.append(
                nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim * n_channels, 64),
                nn.Hardswish(),
                nn.Dropout(p = 0.5),
                nn.Linear(64, 32),
                nn.Hardswish(),
                nn.Dropout(p = 0.5),
                nn.Linear(32, 1),
                nn.ReLU()                
                ))
        self.fc = nn.ModuleList(fc)
        #
    def forward(self, x):
        # Make predictions with each TumE model and concatenate results to feed into new task-specific branches
        x = [conv(x.unsqueeze(1).float()) for conv in self.convolutions]
        x = torch.cat(x, dim = 1)
        x = [fc(x) for fc in self.fc]
        return x    

class Trainer:
    def __init__(self, pretrained_models, viable, task_types = ['r', 'r', 'r', 'r'], learning_rate = 0.001):
        self.model = TransferModel(pretrained_models = [copy.deepcopy(i) for i in pretrained_models])
        # Set network optimizer/loss functions/hyperparameters/, only need to optimize learning rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.f_losses = [InitializeLoss(task_type = i) for i in task_types]
        self.task_types = task_types
        # Set logger
        self.logger = {'seed':[], 'training_loss':[], 'testing_loss':[], 'mean_percentage_error':[], 'number_simulations':[]}
        # TEMULATOR parameter sampler
        self.sampler = Sampler(viable)
        # Track/update TEMULATOR seed to ensure unique simulations over every iteration
        self.seed = 0
        self.nsimulations = 0
         # Normalize labels: mutrate, nclonal, birthtimes, fitness, frequencies
        self.maximum_label_values = [500, 2**np.max(viable['t']), (np.max(viable['b']) - 0.2)/(1-0.2), 1]
        #
    def simulate_step(self, n = 128, parallel = False, n_processors = 20):
        # Parallelize
        if parallel == True:
            pool = multiprocessing.Pool(n_processors)
            # Generate a partial function to run simulator in pooled parallel stream
            partial_temulator = partial(run_temulator, maximum_label_values = self.maximum_label_values, seed = self.seed)
            data = np.array(pool.map(partial_temulator, [self.sampler.get_parameters() for i in range(n)]), dtype=object)
            pool.close()  
            pool.terminate()
            pool.join()
            features, labels = np.array([i[0] for i in data]), np.array([i[1] for i in data])        
        # Run linearly 
        else:
            features, labels = [], []
            for i in range(n):
                print(f'sim {i+1}/{n}')
                parameters = self.sampler.get_parameters()     
                fts, labs = run_temulator(parameters, maximum_label_values = self.maximum_label_values, seed = self.seed)                
                features.append(fts), labels.append(labs)
            features, labels = np.array(features), np.array(labels)        
        # Update number of data points and seed
        self.nsimulations += n
        self.logger['number_simulations'].append(self.nsimulations)        
        self.logger['seed'].append(self.seed)
        self.seed += 1        
        # Structure training set
        train = [torch.Tensor(features),
                 torch.Tensor(labels[:,0]),
                 torch.Tensor(labels[:,1]),
                 torch.Tensor(labels[:,2]),
                 torch.Tensor(labels[:,3])]
        return train, labels
        #
    def train_step(self, batch, batch_number):
        # Train        
        self.model.train()    
        self.optimizer.zero_grad()                
        # Compute loss        
        X, Ys = batch[0], [label for label in batch[1:len(batch)]] # Split features and true targets        
        yhats = self.model(X) # Generates a tuple of all predicted targets            
        cumulative_loss = GetLoss(Ys, yhats, loss_functions = self.f_losses, task_types = self.task_types, device = 'cpu') # For jointly optimizing multiple tasks                                           
        # Backpropagation                    
        cumulative_loss.backward()
        self.optimizer.step()                            
        # Report progress
        batch_size = len(X)                                       
        loss, current = cumulative_loss.item() / batch_size, batch_number * batch_size      
        print(f"loss: {loss :>7f} total_simulations :{self.nsimulations}")  
        return cumulative_loss.item() / batch_size
        #
    def fit(self, batches = 10, batchsize = 128, patience = 3):
        track_labels = []
        early = EarlyStopping(patience = patience)
        for batch_number in range(1, batches + 1):
            print('Batch: ' + str(batch_number))
            # Simulate
            batch, labels = self.simulate_step(n = batchsize)
            #track_labels = track_labels + list(labels)
            # Train
            training_loss = self.train_step(batch, batch_number)
            self.logger['training_loss'].append(training_loss)
            early.update(training_loss) 
            if (early.check() == 1) or (batch_number == batches):
                return self.model, self.logger#, track_labels