import random
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
import os
import pandas as pd
from utils import *

"""
Neural network convolutional multi-task architectures, metrics, and fitting functions
"""

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

def InitializeLoss(task_type: str, posweight = 0.25) -> torch.nn.modules.loss:
    """
    Loss functions for binary classification (b), multi-class classification (m), and regression (r)

    To be more parsimonious for detecting selection, posweight < 1 will increase precision on non-neutral classes
    """
    if task_type == 'b':
        return nn.BCEWithLogitsLoss(pos_weight = torch.Tensor([posweight]))
    elif task_type == 'm':
        return nn.CrossEntropyLoss()
    elif task_type == 'r':
        return nn.L1Loss()
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

def GetAccuracy(Ys: torch.Tensor, yhats: torch.Tensor, task_types: str, threshold = 0.5) -> np.ndarray:
    """
    Computing training metrics for binary classification (b), multi-class classification (m), and regression (r)
    """
    accuracies = []
    for task_ind in range(len(task_types)):
        if len(task_types) == 1:
            yhat, y, task = yhats, Ys[task_ind], task_types[task_ind]
        else:
            yhat, y, task = yhats[task_ind], Ys[task_ind], task_types[task_ind]
        if task == 'b':
            yhat = yhat.view(-1).to(torch.float)
            accuracy = ((torch.sigmoid(yhat).type(torch.float) > threshold) == y).sum().item()
        if task == 'm':
            accuracy = (yhat.argmax(1) == y).type(torch.float).sum().item()
        if task == 'r':            
            accuracy = torch.mean((100 * torch.abs(y - yhat) + 1 / (torch.abs(y) + 1)))     
        accuracies.append(accuracy)
    return np.array(accuracies)    

class ConvBlock(nn.Module):
    """
    A simple skip block across 3 convolutional layers; i.e. reverts to identity if weights deteriorate to zero
    """
    def __init__(self, channels, kernel_size, padding, p = 0.5):
        super(ConvBlock, self).__init__()        
        self.conv_in = nn.Conv1d(in_channels = channels, out_channels = channels, kernel_size = kernel_size, padding = padding)
        self.conv = nn.Conv1d(in_channels = channels, out_channels = channels, kernel_size = kernel_size, padding = padding)
        self.activation = nn.Hardswish()
        self.drop = nn.Dropout(p)
        
    def forward(self, x):
        xa = self.drop(self.activation(self.conv_in(x.float())))
        xa = self.drop(self.activation(self.conv(xa.float())))
        xa = self.drop(self.activation(self.conv(xa.float())))
        xa += x
        return xa

class EvolutionV2(nn.Module):
    """
    Additional architecture with skip connections
    """
    def __init__(self, input_dim = 192, n_out = 4, n_conv = 4, conv_width = [7, 3, 3, 3], drop = 0.5, branch_type = 'Linear', feature_1_dim = 64, feature_2_dim = 128,):
        super(EvolutionV2, self).__init__()
        self.branch_type = branch_type   
        # Convolutional trunk
        conv = []
        conv.append(nn.Conv1d(in_channels = 1, out_channels = n_out, kernel_size = conv_width[0], padding = math.floor(conv_width[0]/2)))
        conv.append(nn.Hardswish())
        conv.append(nn.Dropout(p=0.5))
        for i in range(1,n_conv-1):                
            conv.append(ConvBlock(channels = n_out, kernel_size = conv_width[i], padding = math.floor(conv_width[i]/2)))
            conv.append(nn.Conv1d(in_channels = n_out, out_channels = n_out, kernel_size = 1, padding = 0))
            conv.append(nn.Hardswish())
        #
        conv.append(ConvBlock(channels = n_out, kernel_size = conv_width[-1], padding = math.floor(conv_width[-1]/2)))
        self.conv = nn.Sequential(*conv)
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
                nn.Conv1d(in_channels = n_out, out_channels = n_out, kernel_size = conv_width[-1], padding = math.floor(conv_width[-1]/2)),
                nn.Hardswish(),
                nn.Dropout(p = drop),
                nn.Conv1d(in_channels = n_out, out_channels = 1, kernel_size = conv_width[-1], padding = math.floor(conv_width[-1]/2)),
                nn.Hardswish(),
                nn.AdaptiveAvgPool1d(1)
            )
            # Number of subclones (0, 1, or 2)
            self.nsubclones = nn.Sequential(
                nn.Conv1d(in_channels = n_out, out_channels = n_out, kernel_size = conv_width[-1], padding = math.floor(conv_width[-1]/2)),
                nn.Hardswish(),
                nn.Dropout(p = drop),
                nn.Conv1d(in_channels = n_out, out_channels = 1, kernel_size = conv_width[-1], padding = math.floor(conv_width[-1]/2)),
                nn.Hardswish(),
                nn.AdaptiveAvgPool1d(3)
            )                     
    def forward(self, x):
        x = self.conv(x.unsqueeze(1).float())
        t1 = self.mode(x.float())
        t2 = self.nsubclones(x.float())
        if self.branch_type == 'GAP':
            t1, t2 = t1.squeeze(1), t2.squeeze(1)
        return t1, t2

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

class TwoSubclone(nn.Module):
    """
    Convolutional multi-task architecture to parse frequency and timing of a two subclones
    
    :param input_dim (int): Length of 1-dimensional feature vector
    :param n_out (int): Number of output channels/feature maps for each convolutional layer
    :param n_conv (int): Number of convolutional layers in trunks of network
    :param conv_width (list): Kernel size for convolution layers (where 1st index is for trunk 1, 2nd index is for trunk 2, and 3rd index is for branches if branch_type != 'Linear')
    :param drop (float): Dropout probability following each layer (Default to 0.5 for MC dropout)
    :param branch_type (str): Specifies if output branches should pass through fully-connected (Linear) or global average pooling (Default 'Linear')
    """
    def __init__(self, input_dim = 448, n_out = 4, n_conv = 4, conv_width = [3, 7, 11], drop = 0.5, branch_type = 'Linear'):
        super(TwoSubclone, self).__init__() 
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
            self.frequency2 = nn.Sequential(            
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
            self.timing2 = nn.Sequential(            
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
            self.frequency2 = nn.Sequential(            
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
            self.timing2 = nn.Sequential(            
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
        t20, t21 = self.frequency2(x1.float()), self.frequency2(x2.float())
        t30, t31 = self.timing1(x1.float()), self.timing1(x2.float())
        t40, t41 = self.timing2(x1.float()), self.timing2(x2.float())        
        if self.branch_type == 'Linear':
            t1 = torch.mean(torch.cat((t10, t11), dim = 1), dim=1)
            t2 = torch.mean(torch.cat((t20, t21), dim = 1), dim=1)
            t3 = torch.mean(torch.cat((t30, t31), dim = 1), dim=1)
            t4 = torch.mean(torch.cat((t40, t41), dim = 1), dim=1)
            return t1.unsqueeze(1), t2.unsqueeze(1), t3.unsqueeze(1), t4.unsqueeze(1)
        else:
            t1 = torch.mean(torch.cat((t10, t11), dim = 1), dim=1)
            t2 = torch.mean(torch.cat((t20, t21), dim = 1), dim=1)
            t3 = torch.mean(torch.cat((t30, t31), dim = 1), dim=1)
            t4 = torch.mean(torch.cat((t40, t41), dim = 1), dim=1)
            return t1, t2, t3, t4

class TransferModel(nn.Module):
    """
    Takes pre-trained TumE models and mutates network for transfer learning on TEMULATOR simulations
    
    :param pretrained_models (list): A list of TumE models, takes each model and uses it as a tunable feature extractor
    :input_dim (int): The input feature width
    """
    def __init__(self, pretrained_models, n_linear = 4, gradients = True, input_dim = 192, n_tasks = 4):
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
        # Specify if gradients in pre-trained models should be updated or not during training
        if gradients == False:
            for param in self.convolutions.parameters():
                param.requires_grad = False
        #
        # Build fully connected layers
        n_channels = np.sum(nchannels)
        n_linear = [32*i for i in range(1, n_linear+1)][::-1]
        #
        # Build fully-connected branches for each task
        fcs = []
        for i in range(n_tasks):
            fc = []
            fc.append(nn.Flatten())
            fc.append(nn.Linear(input_dim * n_channels, n_linear[0]))
            fc.append(nn.Hardswish())
            fc.append(nn.Dropout(p = 0.5))
            if len(n_linear) > 1:
                for i in range(1, len(n_linear)):
                    fc.append(nn.Linear(n_linear[i-1], n_linear[i]))
                    fc.append(nn.Hardswish())
                    fc.append(nn.Dropout(p=0.5))
            fc.append(nn.Linear(32, 1))
            fc.append(nn.ReLU())
            fcs.append(fc)
        # Task-specific branches
        self.fc1 = nn.Sequential(*fcs[0])
        self.fc2 = nn.Sequential(*fcs[1])
        self.fc3 = nn.Sequential(*fcs[2])
        self.fc4 = nn.Sequential(*fcs[3])
        #
    def forward(self, x):
        # Make predictions with each TumE model and concatenate results to feed into new task-specific branches
        x = [conv(x.unsqueeze(1).float()) for conv in self.convolutions]
        x = torch.cat(x, dim = 1)
        x1, x2, x3, x4 = self.fc1(x), self.fc2(x), self.fc3(x), self.fc4(x)
        return x1, x2, x3, x4

class EvoNet:
    """
    An automated initialization and model fitting class for multi-task PyTorch models

    :param model (nn.Module): A pytorch neural network module
    :param task_types (list): A list of strings specifiy each task type (b = binary classification, m = multiclass, r = regression)
    """
    def __init__(self, model: nn.Module, task_types: list, posweight = 0.25, device = None) -> None:                        
        if device == 'cpu': # If you want to run on CPU even if GPU is present
            self.device = torch.device('cpu')
            self.model = model
        
        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if self.device.type == 'cpu':
                self.model = model    
            if torch.cuda.device_count() == 1:
                self.model = model.to('cuda', non_blocking = True)
            if torch.cuda.device_count() > 1:
                print("Using", torch.cuda.device_count(), "GPUs")
                devices = [i for i in range(torch.cuda.device_count())]
                self.model = nn.DataParallel(model.to('cuda', non_blocking = True), device_ids = devices)        
                
        self.task_types = task_types
    
        # Generate array of loss functions
        self.f_losses = [InitializeLoss(task_type = i, posweight = posweight) for i in task_types]
        
    def iterativefit(self, directory: str, loading_function, epochs = 10, optimizer = 'Adam', learning_rate = 1e-3, batchsize = 2048, patience = 10, delta = 0, momentum = 0.9, report_progress = True):
        """
        Fits a pytorch neural network module by iteratively loading data from a specified directory

        :param directory (str): The path to directory containing data used for training
        :param loading_function: A loading function that reads and processes data from specified directory
        :param epochs (int): Maximum number of epochs to train for
        :param optimizer (str): Name of optimizer to use for updating weights (SGD, RMSprop, or Adam)
        :param learning_rate (float): Learning rate for optimizer
        :param batchsize (int): Size of batches for each weight update
        :param patience (int): Number of epochs without a change in loss before terminating training
        :param delta (int): The required change in loss to consider an epoch as not improved for early stopping
        :param momentum (float): If the optimizer is SGD, momentum will be set
        """
        rmsprop, sgd, adam = torch.optim.RMSprop(self.model.parameters(), lr = learning_rate), torch.optim.SGD(self.model.parameters(), lr = learning_rate, momentum = momentum), torch.optim.Adam(self.model.parameters(), lr = learning_rate)
        if optimizer == 'SGD': optimizer = sgd
        if optimizer == 'RMSprop': optimizer = rmsprop
        if optimizer == 'Adam': optimizer = adam        
        
        # Early Stopping
        early = EarlyStopping(patience = patience, delta = delta)        

        # Fit model
        training_loss, testing_loss, accuracies = [], [], []
        for epoch in range(1, epochs + 1):

            # Setup file loading and test set for iterative training
            files, nfiles, counter = sorted(os.listdir(directory)), len(os.listdir(directory)), 0
            files, test_file = files[0:len(files)-1], files[len(files)-1] # Ensuring test set remains independent
            np.random.shuffle(files)
            train_size, total_loss = 0, 0
            self.test, testset = loading_function(directory, test_file, batchsize = batchsize, testset = True)            
            
            # Iterate across all files in directory
            for file in files:
                counter += 1
                print(f"Iterating over file {counter} out of file {nfiles-1}")
                self.train = loading_function(directory, file, batchsize = batchsize)
                
                # Training step                            
                train_size, total_loss = len(self.train.dataset), 0
                self.model.train()

                # Run batches for training
                for batch_number, batch_data in enumerate(self.train):
                    optimizer.zero_grad()
                    X, Ys = batch_data[0].to(self.device, non_blocking = True), [label.to(self.device, non_blocking = True) for label in batch_data[1:len(batch_data)]] # Split features and true targets        
                    yhats = self.model(X) # Generates a tuple of all predicted targets            
                    cumulative_loss = GetLoss(Ys, yhats, loss_functions = self.f_losses, task_types = self.task_types, device = self.device) # For jointly optimizing multiple tasks                                           
                    
                    # Backpropagation                    
                    cumulative_loss.backward()
                    optimizer.step()            
                    total_loss += cumulative_loss.item()                   
                    
                    # Report progress
                    batch_size = len(X)
                    if int(round(0.1 * (train_size/batch_size), 1 - len(str(int(0.1 * (train_size/batch_size)))))) != 0:                                 
                        if batch_number % int(round(0.1 * (train_size/batch_size), 1 - len(str(int(0.1 * (train_size/batch_size)))))) == 0:
                            if report_progress == True:
                                loss, current = cumulative_loss.item() / len(X), batch_number * len(X)      
                                print(f"loss: {cumulative_loss.item() :>7f}  [{current:>5d}/{train_size:>5d}]")
                
                # Check GPU utilization if running on a cuda                                    
                if self.device.type == 'cuda':
                    for dev in range(torch.cuda.device_count()):
                        print(torch.cuda.get_device_name(dev))
                        print('Memory Usage:')
                        print('Allocated:', round(torch.cuda.memory_allocated(dev)/1024**3,1), 'GB')
                        print('Cached:   ', round(torch.cuda.memory_reserved(dev)/1024**3,1), 'GB')
                
                # Store training loss and update early stopping                                        
                training_loss.append(total_loss / train_size)
                early.update(total_loss / train_size)                             
                print(f"File {file} with average training loss of: {total_loss/train_size}")

                # Testing
                total_loss, correct = 0, torch.zeros(len(self.task_types))
                test_size = len(self.test.dataset)            
                self.model.eval()
                with torch.no_grad():

                    # Run batches for testing
                    for batch_data in self.test:                    
                        X, Ys = batch_data[0].to(self.device, non_blocking = True), [label.to(self.device, non_blocking = True) for label in batch_data[1:len(batch_data)]] 
                        yhats = self.model(X)
                        total_loss += GetLoss(Ys, yhats, loss_functions = self.f_losses, task_types = self.task_types, device = self.device) # For jointly optimizing multiple tasks                           
                        correct += GetAccuracy(Ys, yhats, self.task_types)                        
                    correct /= test_size 
                    total_loss /= test_size
                    accuracies.append(correct)
                    if type(total_loss) == type(float()):
                        testing_loss.append(total_loss)
                    else:
                        testing_loss.append(total_loss.item())

                    # If all tasks are regression compute extra metrics to check for generalization across entire input range                                        
                    if np.all(np.array(self.task_types) == 'r') == True:
                        report = [f"Test> Avg. loss: {total_loss:>10f} | "] + [f"Task {i+1} metric: {(correct[i]):>0.5f}% | " for i in range(len(self.task_types))]
                        print(''.join(report))
                        pred = self.model(torch.Tensor(testset))
                        print(''.join([f"Task {i+1} var pred.: {(torch.var(pred[i]).item()):>0.5f} | " for i in range(len(self.task_types))]))
                        print(''.join([f"Task {i+1} max pred.: {(torch.max(pred[i]).item()):>0.5f} | " for i in range(len(self.task_types))]))
                        print(''.join([f"Task {i+1} min pred.: {(torch.min(pred[i]).item()):>0.5f} | " for i in range(len(self.task_types))]))
                    else:
                        report = [f"Test> Avg. loss: {total_loss:>10f} | "] + [f"Task {i+1} metric: {(100*correct[i]):>0.5f}% | " for i in range(len(self.task_types))]
                        print(''.join(report))

                # Check for early stopping
                if (early.check() == 1) or (epoch == epochs):
                    # Save output dataframe
                    df = pd.DataFrame({'epoch':[i+1 for i in range(len(training_loss))], 'training_loss':training_loss, 'testing_loss':testing_loss})
                    pred = self.model(torch.Tensor(testset))
                    for i in range(len(self.task_types)):
                        df["task_" + str(i+1) + "_metric"] = [j[i].item() for j in accuracies]
                        if np.all(np.array(self.task_types) == 'r') == True:
                            df["task_" + str(i+1) + "_var"] = torch.var(pred[i]).item()                        
                            df["task_" + str(i+1) + "_max"] = torch.max(pred[i]).item()
                            df["task_" + str(i+1) + "_min"] = torch.min(pred[i]).item()
                    return self.model, df
            print(f"Epoch {epoch} with average training loss of: {total_loss/train_size}")

def load_model(directory: str, mod: str, kind: str, input_dim = 192):
    """
    Load a model with standardized naming conventions
    
    :param directory (str): The path where models are stored
    :param mod (str): The name of the trained pytorch model
    :param kind (str): The name of the inference task the model is used for
    :param input_dim (int): Width of input feature vector
    """
    if kind != 'evolution':
        name, n_out, n_conv, branch_type, conv_width1, conv_width2, conv_width3, lr, extra = mod.split('_')
    else:
        name, n_out, n_conv, branch_type, conv_width1, conv_width2, conv_width3, lr, patience, extra = mod.split('_')    
    if kind == 'evolution':
        m = Evolution(input_dim = int(input_dim), n_out = int(n_out), n_conv = int(n_conv), branch_type = branch_type, conv_width = [int(conv_width1), int(conv_width2), int(conv_width3)], drop = 0.5)
    if kind == 'onesubclone':
        m = OneSubclone(input_dim = int(input_dim), n_out = int(n_out), n_conv = int(n_conv), branch_type = branch_type, conv_width = [int(conv_width1), int(conv_width2), int(conv_width3)], drop = 0.5)
    if kind == 'twosubclone':
        m = TwoSubclone(input_dim = int(input_dim), n_out = int(n_out), n_conv = int(n_conv), branch_type = branch_type, conv_width = [int(conv_width1), int(conv_width2), int(conv_width3)], drop = 0.5)
    m.load_state_dict(torch.load(directory + mod))
    m.eval()
    return m

def prediction(directory: str, mod: str, kind: str, montecarlo = 50, means = True, path = None, vaf = None, dp = None):
    """
    A basic prediction function for evaluating deep learning models

    :param directory (str): The path where models are store that must end with /
    :param mod (str): The name of the trained pytorch model
    :param kind (str): The name of the inference task the model is used for
    :param montecarlo (int): The number of stochastic passes to run through the network when generating uncertainty estimates
    :param means (bool): If True, only return mean estimates; if False, returns an array of N montecarlo predictions
    :param path (str): If None, then inferences are made using vaf and dp input; if str, then the predictions will be made on all files in path
    :param vaf (np.ndarray): VAF for all mutations in diploid regions of a given sample
    :param dp (np.ndarray or int): The sequencing depth for each mutation or the mean depth of the sample
    """
    model = load_model(directory = directory, mod = mod, kind = kind)
    model.train()
    
    if path != None:
        data = np.vstack(np.load(path, allow_pickle=True))
        features, _, _, _, _, _, _ = sim2evo(data)
    if type(vaf) != type(None):
        features = [np.hstack(vaf2feature(vaf, depth = dp))]
    
    if kind == 'evolution':
        mcdropout = [model(torch.Tensor(features)) for i in range(montecarlo)]
        mode = np.array([torch.sigmoid(i[0]).detach().numpy().flatten() for i in mcdropout])
        nsubs = np.array([F.softmax(i[1], dim = 1).detach().numpy() for i in mcdropout])
        if means == True:
            mode = np.mean(mode, axis = 0)
            nsubs = np.mean(nsubs, axis = 0)
            return mode, nsubs
        else:
            if type(vaf) != type(None):
                nsubs = np.array([i[0] for i in nsubs])
            return mode, nsubs
    
    if kind == 'onesubclone':
        mcdropout = [model(torch.Tensor(features)) for i in range(montecarlo)]
        frequency = np.array([rescale_frequencies(i[0].detach().numpy().flatten(),direction='predict') for i in mcdropout])
        time = np.array([i[1].detach().numpy().flatten() for i in mcdropout])
        if means == True:
            frequency = np.mean(frequency, axis = 0)
            time = np.mean(time, axis = 0)
            return frequency, time
        else:
            if type(vaf) != type(None):
                frequency, time = frequency.flatten(), time.flatten()
            return frequency, time
    
    if kind == 'twosubclone':
        mcdropout = [model(torch.Tensor(features)) for i in range(montecarlo)]
        frequency1 = np.array([rescale_frequencies(i[0].detach().numpy().flatten(), direction='predict') for i in mcdropout])
        frequency2 = np.array([rescale_frequencies(i[1].detach().numpy().flatten(), direction='predict') for i in mcdropout])
        time1 = np.array([i[2].detach().numpy().flatten() for i in mcdropout])
        time2 = np.array([i[3].detach().numpy().flatten() for i in mcdropout])
        if means == True:
            frequency1 = np.mean(frequency1, axis = 0)
            frequency2 = np.mean(frequency2, axis = 0)
            time1 = np.mean(time1, axis = 0)
            time2 = np.mean(time2, axis = 0)
            return frequency1, frequency2, time1, time2
        else:
            if type(vaf) != type(None):
                frequency1, frequency2, time1, time2 = frequency1.flatten(), frequency2.flatten(), time1.flatten(), time2.flatten()
            return frequency1, frequency2, time1, time2