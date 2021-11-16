import importlib.resources
from joblib import load
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn import linear_model
from .utils import *
from .nets import *

"""
TumE predictions
"""

def load_model(mod: str, kind: str, input_dim = 192):
    """
    Loads a TumE PyTorch model state dictionary

    :param mod (str): The name of pytorch model in standardized format (see line 1 split method for structure)
    :param kind (str): One of evolution, onesubclone, or twosubclone
    :param input_dim (int): The width of feature vector
    :param external (bool): If False, models will be loaded from internal TumE library, else mod references a direct path to a pytorch model
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
    mod = importlib.resources.open_binary('TumE.models', mod)
    m.load_state_dict(torch.load(mod))
    m.eval()
    return m

def transfer_model(input_dim = 192):
    # Initialize architecture
    mod = '7_5.444791546772337e-05_True_True.NCCH3Y8VJCWA9MZ.pt'
    n_linear = int(mod.split('_')[0])
    MS = 'evolution_11_5_Linear_17_9_7_9.027991854570908e-06_5_0.14.TASYG7N3IJR1DLN.pt'
    OS = 'onesubclone_15_11_Linear_15_3_7_0.0001698628401328024_6.GS3BEXB3O906DHE.pt'
    model1 = load_model(mod = MS, kind = 'evolution')
    model2 = load_model(mod = OS, kind = 'onesubclone')
    pretrained_models = [model1, model2]
    m = TransferModel(pretrained_models, n_linear = n_linear, gradients = True, input_dim = input_dim, n_tasks = 4)
    mod = importlib.resources.open_binary('TumE.transfer_models', mod)
    m.load_state_dict(torch.load(mod))
    m.eval()
    return m

def prediction(mod: str, kind: str, montecarlo = 50, means = True, path = None, vaf = None, dp = None, seed = 456789) -> tuple:
    """
    Generates cancer evolution predictions given a VAF distribution or a large set of VAF distributions in numpy format

    Example for single sample inference:
        >> prediction(vaf = diploid_mutation_vafs,
                      directory = '../models/', 
                      mod = 'evolution_16_6_Linear_15_1_9_0.00019152200420744457_4.FKGLFDJPGY7OBMR',
                      kind = 'evolution',
                      montecarlo = 250,
                      means = False)

    :param directory (str): The directory containing the PyTorch deep learning models
    :param mod (str): The name of the model in a structured format (e.g kind_nout_nconv_branchtype_cwidth1_cwidth2_cwidth3_lr_*.pt, see load_model or ./models directory)
    :param kind (str): The type of model to make inferences with (one of evolution, onesubclone, or twosubclone)
    :param montecarlo (int): The number of stochastic passes through the network for prediction
    :param means (bool): If true, only the mean predictions are returned for a given sample. If false, the entire prediction distribution is returned
    :param path (NoneType): If inferences are made in bulk, path specifies the location of numpy file containing N pre-processed TumE features (generally only used for model development/testing)
    :param vaf (NoneType): If inferences are made per sample, then vaf is a numpy array of containing all VAFs found in diploid regions of copy number corrected samples
    :param external (bool): If False, models will be loaded from internal TumE library, else mod references a direct path to a pytorch model

    >return (tuple) Predictions for a given inference task

    Notes:
        - If kind == evolution then tuple = (mode, nsub)
        - If kind == onesubclone then tuple = (frequency, time)
        - If kind == twosubclone then tuple = (frequency1, frequency2, time1, time2)

    Development notes:
        - Could probably remove some of the old variables used for running jobs on the HPC
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load model
    model = load_model(mod = mod, kind = kind)
    model.train()
    with torch.no_grad():
        # Bulk inference in pre-processed/feature engineered numpy arrays
        if path != None:
            data = np.vstack(np.load(path, allow_pickle=True))
            features, _, _, _, _, _, _ = sim2evo(data)
        
        # Single sample inference
        if type(vaf) != type(None):
            features = [np.hstack(vaf2feature(vaf, depth = dp))]
        
        # Predicting evolutionary mode and number of subclones
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
        
        # Predicting the frequency and emergence time of one subclone
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
        
        # Predicting the frequency and emergence time of two subclones
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


def binomial_bounds(freq, depth, nvar = 2):
    """
    Returns the upper and lower VAF cutoffs based on 2 or 3 binomial standard deviations
    """
    f_low = freq - ( ( nvar*np.sqrt(freq*depth*(1-(freq))) ) / depth)
    f_high = freq + ( ( nvar*np.sqrt(freq*depth*(1-(freq))) ) / depth)
    return f_low, f_high

def estimate(df, vaf_name = 'VAF', dp_name = 'DP', nmc = 100, clustering = 'binomial', silent = False, nvar = 2, seed = 123456) -> dict:
    """
    Standard prediction method for TumE
    
    Input dataframe must have VAF and depth information

    :param vaf (np.ndarray): Purity-corrected VAFs from diploid regions
    :param dp (int or np.ndarray): Mean sequencing depth or an array of sequencing depth at each position
    :param nmc (int): Number of stochastic passes through network used for prediction 
    :param eqint (list): Equal-tailed interval where 1st and 2nd indices are the lower and upper quantiles
    
    >return (list) Cancer evolution predictions for all TumE models (evolution, onesubclone, twosubclone)
    
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load VAF and DP information
    df = df.sort_values(by = vaf_name).reset_index()
    df = df[(df[vaf_name] > 0) & (df[vaf_name] < 1)]
    vaf = np.array(list(df[vaf_name]))
    dp = np.array(list(df[dp_name]))
    
    # Process
    dp = np.mean(dp)

    # Models
    MS = 'evolution_11_5_Linear_17_9_7_9.027991854570908e-06_5_0.14.TASYG7N3IJR1DLN.pt'
    OS = 'onesubclone_30_2_Linear_5_11_17_0.00011838209194147751_6.QME74JJO18RFEL3.pt'
    TS = 'twosubclone_30_14_Linear_3_11_13_0.00014449996176045516_5.PDBHTZAX308BPA1.pt'
    mods = [MS, OS, TS]

    # Predictions
    m, ns = prediction(mod = mods[0], kind = 'evolution', montecarlo = nmc, means = False, vaf = vaf, dp = dp)
    f, t = prediction(mod = mods[1], kind = 'onesubclone', montecarlo = nmc, means = False, vaf = vaf, dp = dp)
    f1, f2, t1, t2 = prediction(mod = mods[2], kind = 'twosubclone', montecarlo = nmc, means = False, vaf = vaf, dp = dp)

    # Get mean predictions and intervals
    nsub_mean, nsub_lower, nsub_upper = [], [], []
    for i in range(3):
        nsub_mean.append(round(np.mean(ns, axis=0)[i], 3))
        nsub_lower.append(round(np.quantile(ns, q=[0.055],axis=0)[0][i],3))
        nsub_upper.append(round(np.quantile(ns, q=[0.945],axis=0)[0][i],3))
    
    # Parsimony check for 1 and 2 subclones (overlapping intervals)
    if (np.round(np.quantile(ns, q = 0.055, axis=0)[2] - np.quantile(ns, q = 0.945, axis=0)[1],2) <= 0) & (np.quantile(m.flatten(), q=0.055) > 0.5):
        explain_with_parsimony = True
    else:
        explain_with_parsimony = False

    # Print output
    if silent == False:
        from termcolor import colored
        print(colored("------------------------------------------------------------------------", 'blue'))
        print(colored("TumE: Cancer evolution inference with synthetic supervised deep learning", 'blue'))
        print(colored("------------------------------------------------------------------------", 'blue'))    
        print("")
        if np.round(dp, 2) < 70:
            print(f"Warning! Sequencing depth of {np.round(dp, 2)} is low (< 70x) and not optimal for prediction. Please inspect outputs. Greater than >>70x is recommended.")
        print("")
        print(colored("Evolutionary mode prediction:", 'magenta'))
        print(f'   P(selection) {"{:.3f}".format(np.mean(m))} [{round(np.quantile(m, q=[0.055])[0],3)}, {round(np.quantile(m, q=[0.945])[0],3)}]')
        print("")
        print(colored('>>> Detect positive selection!', 'green')) if (np.quantile(m, q=0.055) > 0.5) else print(colored('>>> Detect neutral evolution! Lower bound of 89% interval < 0.5', 'green'))
        print("")                               
        if (np.quantile(m, q=0.055) > 0.5):
            print(colored("Number of subclone prediction:", 'magenta'))
            print(f'   P(0 subclone) {"{:.3f}".format(nsub_mean[0])} [{nsub_lower[0]}, {nsub_upper[0]}]')
            print(f'   P(1 subclone) {"{:.3f}".format(nsub_mean[1])} [{nsub_lower[1]}, {nsub_upper[1]}]')
            print(f'   P(2 subclone) {"{:.3f}".format(nsub_mean[2])} [{nsub_lower[2]}, {nsub_upper[2]}]')
            print("")
            print(colored(f'>>> Detect {np.argmax(nsub_mean)} subclone!', 'green')) if np.argmax(nsub_mean) == 1 else print(colored(f'>>> Detect {np.argmax(nsub_mean)} subclones!', 'green'))
            print("")
            print(colored('Subclone parameter estimates', 'magenta'))
            if ((np.argmax(nsub_mean) == 1) or (explain_with_parsimony == True)):
                print('   1 subclone:')
                print(f'      - frequency (cellular fraction / 2): {"{:.3f}".format(np.mean(f))} [{round(np.quantile(f, q=[0.055])[0],3)}, {round(np.quantile(f, q=[0.945])[0],3)}]')
                print('')
                # Scale time as time was measured as a harmonic series with respect to population size i.e ratio of sum(1/n) / sum(1/N) for i = 1 to n or N
                scale_time = lambda t: np.log2(((np.exp(t)*np.sum([1/i for i in range(1, 1000)])) * (1e10/1e3)))                
                print(colored('Parameter estimates in development/less certainty:', 'red'))
                print(f'      - emergence time (proportion of tumour doublings since inception): {"{:.3f}".format(scale_time(np.mean(t)))} [{round(scale_time(np.quantile(t, q=[0.055])[0]),3)}, {round(scale_time(np.quantile(t, q=[0.945])[0]),3)}]')
                print("")
            if ((np.argmax(nsub_mean) == 2) & (explain_with_parsimony == False)):
                print('   2 subclones:')
                print(f'      - frequency 1 (cellular fraction / 2): {"{:.3f}".format(np.mean(f1))} [{round(np.quantile(f1, q=[0.055])[0],3)}, {round(np.quantile(f1, q=[0.945])[0],3)}]')
                print(f'      - frequency 2 (cellular fraction / 2): {"{:.3f}".format(np.mean(f2))} [{round(np.quantile(f2, q=[0.055])[0],3)}, {round(np.quantile(f2, q=[0.945])[0],3)}]')
                print('')
                # Scale time as time was measured as a harmonic series with respect to population size i.e ratio of sum(1/n) / sum(1/N) for i = 1 to n or N
                scale_time = lambda t: np.log2(((np.exp(t)*np.sum([1/i for i in range(1, 1000)])) * (1e10/1e3)))
                print(colored('Parameter estimates in development/less certainty:', 'red'))
                print(f'      - emergence time 1 (proportion of tumour doublings since inception): {"{:.3f}".format(scale_time(np.mean(t1)))} [{round(scale_time(np.quantile(t1, q=[0.055])[0]),3)}, {round(scale_time(np.quantile(t1, q=[0.945])[0]),3)}]')
                print(f'      - emergence time 2 (proportion of tumour doublings since inception): {"{:.3f}".format(scale_time(np.mean(t2)))} [{round(scale_time(np.quantile(t2, q=[0.055])[0]),3)}, {round(scale_time(np.quantile(t2, q=[0.945])[0]),3)}]')
    
    # Segment mutations
    annotation = ['' for i in range(len(vaf))]        
    if ((np.argmax(nsub_mean) == 1) & (np.quantile(m, q=0.055) > 0.5)) or (explain_with_parsimony == True):
        f_low, f_high = binomial_bounds(np.mean(f), dp, nvar = nvar)
        neutral = ['Neutral tail' for i in np.sort(vaf) if i < f_low]
        subclone = ['Subclone' for i in np.sort(vaf) if (i >= f_low) & (i <= f_high)]
        clonal = ['Clonal' for i in np.sort(vaf) if i > f_high]
        if clustering == 'binomial':
            annotation = neutral + subclone + clonal
        elif clustering == 'gaussian':
            gm = GaussianMixture(n_components=3, random_state=0, means_init=[[2/dp], [np.mean(f)], [0.5]]).fit_predict(np.sort(vaf).reshape(-1,1))
            annotation = gm
            annotation = [annotation[i] if np.sort(vaf)[i] < 0.55 else 2 for i in range(len(annotation))]
            annotation = ['Neutral tail' if i == 0 else i for i in annotation]
            annotation = ['Subclone' if i == 1 else i for i in annotation]
            annotation = ['Clonal' if i == 2 else i for i in annotation]
        else:
            raise ValueError('Clustering type must be either binomial or gaussian')

    elif (np.argmax(nsub_mean) == 2) & (np.quantile(m, q=0.055) > 0.5):
        f1_low, f1_high = binomial_bounds(np.mean(f1), dp, nvar = nvar)
        f2_low, f2_high = binomial_bounds(np.mean(f2), dp, nvar = nvar)
        neutral1 = ['Neutral tail' for i in np.sort(vaf) if i < f1_low]
        subcloneB = ['Subclone B' for i in np.sort(vaf) if (i >= f1_low) & (i <= f1_high + (f2_low - f1_high)/2)]        
        subcloneA = ['Subclone A' for i in np.sort(vaf) if (i >= f2_low - (f2_low - f1_high)/2) & (i <= f2_high)]
        clonal = ['Clonal' for i in np.sort(vaf) if i > f2_high]
        if clustering == 'binomial':
            annotation = neutral1 + subcloneB + subcloneA + clonal
        elif clustering == 'gaussian':
            gm = GaussianMixture(n_components=4, random_state=0, means_init=[[2/dp], [np.mean(f1)], [np.mean(f2)], [0.5]]).fit_predict(np.sort(vaf).reshape(-1,1))
            annotation = gm
            annotation = [annotation[i] if np.sort(vaf)[i] < 0.55 else 3 for i in range(len(annotation))]
            annotation = ['Neutral tail' if i == 0 else i for i in annotation]
            annotation = ['Subclone B' if i == 1 else i for i in annotation]
            annotation = ['Subclone A' if i == 2 else i for i in annotation]
            annotation = ['Clonal' if i == 3 else i for i in annotation]
        else:
            raise ValueError('Clustering type must be either binomial or gaussian')
    else:                
        gm = GaussianMixture(n_components=2, random_state=0, means_init=[[0.1], [0.5]]).fit_predict(np.sort(vaf).reshape(-1,1))
        annotation = gm
        annotation = [annotation[i] if np.sort(vaf)[i] < 0.55 else 1 for i in range(len(annotation))]
        annotation = ['Neutral tail' if i == 0 else i for i in annotation]
        annotation = ['Clonal' if i == 1 else i for i in annotation]

    # Output dataframe for predictions
    p_selection = (np.mean(m), np.quantile(m, q=[0.055])[0], np.quantile(m, q=[0.945])[0])
    p_0subclone = (nsub_mean[0], nsub_lower[0], nsub_upper[0])
    p_1subclone = (nsub_mean[1], nsub_lower[1], nsub_upper[1])
    p_2subclone = (nsub_mean[2], nsub_lower[2], nsub_upper[2])
    f_1subclone = (np.mean(f), np.min(f), np.max(f))
    f_2subclone_1 = (np.mean(f1), np.min(f1), np.max(f1))
    f_2subclone_2 = (np.mean(f2), np.min(f2), np.max(f2))
    mode_prediction = 'selection' if p_selection[1] > 0.5 else 'neutral'
    if ((np.argmax(nsub_mean) > 0) & (explain_with_parsimony == True) & (mode_prediction == 'selection')):
        nsub_prediction = 1
    elif ((np.argmax(nsub_mean) == 2) & (explain_with_parsimony == False) & (mode_prediction == 'selection')):
        nsub_prediction = 2
    else:
        nsub_prediction = 0
    output = pd.DataFrame({
        'Mode': [mode_prediction], 'n_subclones': [nsub_prediction],
        'P(Selection)':p_selection[0], 'L89% P(Selection)':p_selection[1], 'U89% P(Selection)':p_selection[2],
        'P(0 subclone)':p_0subclone[0], 'L89% P(0 subclone)':p_0subclone[1], 'U89% P(0 subclone)':p_0subclone[2],
        'P(1 subclone)':p_1subclone[0], 'L89% P(1 subclone)':p_1subclone[1], 'U89% P(1 subclone)':p_1subclone[2],
        'P(2 subclone)':p_2subclone[0], 'L89% P(2 subclone)':p_2subclone[1], 'U89% P(2 subclone)':p_2subclone[2],
        'freq_1_sub':f_1subclone[0], 'min_freq_1_sub':f_1subclone[1], 'max_freq_1_sub':f_1subclone[2],
        'freq_2_sub_B':f_2subclone_1[0], 'min_freq_2_sub_B':f_2subclone_1[1], 'max_freq_2_sub_B':f_2subclone_1[2],
        'freq_2_sub_A':f_2subclone_2[0], 'min_freq_2_sub_A':f_2subclone_2[1], 'max_freq_2_sub_A':f_2subclone_2[2],
    })

    # Add annotations to input data frame
    df['annotation'] = annotation
    if vaf_name != 'VAF':        
        df['VAF'] = df[vaf_name]
        df = df.drop(columns = [vaf_name])
    if dp_name != 'DP':
        df['DP'] = df[dp_name]
        df = df.drop(columns = [dp_name])

    return {'all_estimates':[m, ns, f, t, f1, f2, t1, t2], 
            'predictions': output,
            'annotated': df}

def transfer_predictions(features, nmc=50, scaled_popsize = 10e8, genomesize = 3.1e9, seed = 123456, constrain_mc = True, mutrate_correction = True) -> tuple:
    """
    Mutation rate, subclone emergence time, subclone fitness, and subclone frequency estimates         
    
    Notes:
     - Mutation rate can be estimated in any neutral or non-neutral tumour
     - Subclone parameters can be estimated in samples with a single subclone
     - Rescale metrics as values were scaled to 0-1 during model training => [500, np.log2(8192), 12.25, 1] Mutrate, time, fitness, frequency    
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if constrain_mc == True:
        mod = transfer_model()
        mod.train()
        mod.convolutions.eval()

    # Mutation rate correction    
    mutrate_correction = load(importlib.resources.open_binary('TumE.transfer_models', 'mutrate_linear.joblib'))
    
    # Make estimates
    with torch.no_grad():
        mcdropout = [mod(torch.Tensor([features])) for i in range(nmc)]

    mutrate = np.array([i[0].detach().numpy().flatten() for i in mcdropout]).flatten()
    time = np.array([i[1].detach().numpy().flatten() for i in mcdropout]).flatten()
    fitness = np.array([i[2].detach().numpy().flatten() for i in mcdropout]).flatten()
    frequency = np.array([i[3].detach().numpy().flatten() for i in mcdropout]).flatten()    

    with np.errstate(divide='ignore', invalid='ignore'): # For cases when t or f == 0, nonfinite/zeros removed in temulator_estimate
        # Rescale values from 0-1 back into original range
        rescale = [500, 8192, 12.25, 1]
        mutrate, time, fitness, frequency = mutrate*rescale[0], np.log2(time*rescale[1]), fitness*rescale[2], frequency*rescale[3]
        
        # Apply the post-hoc linear correction to mutation rate that addresses systematic overestimation
        if mutrate_correction == True:
            mutrate = mutrate_correction.predict(mutrate.reshape(-1,1))
        
        # Convert mutation rate into per base estimates
        mutrate = mutrate / genomesize

        # Update time with stochastic correction as outlined in Williams et al. 2018/Durret
        time = time - np.euler_gamma/np.log(2)

        # Rescale time in tumour doublings and fitness to realistic population size        
        fitness = 1 + (fitness-1)*(np.log2((1-frequency)*1e4)-time) / (np.log2((1-frequency)*scaled_popsize)-time*np.log2(1e8)/np.log2(1e4))    
        
    # Rescale time back to scaled_population_size
    time = time*np.log(scaled_popsize)/np.log(1e4)

    return (mutrate, time, fitness, frequency)


def temulator_estimates(df, vaf_name='VAF', dp_name='DP', nmc=50, scaled_popsize=10e8, genomesize = 1, seed = 123456) -> tuple:
    """
    Estimating TEMULATOR subclone parameters in patient tumours   
    
    Notes:
    - Estimates based models trained on TEMULATOR synthetic data with pre-trained TumE model
      (Ref: TEMULATOR simulation framework built by Heide et al. 2018)
    """
    # Load VAF and DP information
    df = df.sort_values(by = vaf_name).reset_index()
    df = df[(df[vaf_name] > 0) & (df[vaf_name] < 1)]
    vaf = np.array(list(df[vaf_name]))
    dp = np.array(list(df[dp_name]))

    # Convert VAF to features
    features = np.hstack(vaf2feature(vaf, depth = dp))
    
    # Estimates
    mutrate, time, fitness, frequency = transfer_predictions(features, nmc=nmc, scaled_popsize=scaled_popsize, genomesize=genomesize, seed=seed)
    
    # Remove any non-finite or zeros
    mutrate = mutrate[(np.isfinite(mutrate)) & (mutrate > 0)]
    time = time[(np.isfinite(time)) & (time > 0)]
    fitness = fitness[(np.isfinite(fitness)) & (fitness > 0)] 
    frequency = frequency[(np.isfinite(frequency)) & (frequency > 0)] 

    # MC estimates can be degenerate for very low mutation rate samples
    # Hence, if the mutrate vector length is == 0 then re-run mutation rate analysis without constraint
    while len(mutrate) < np.min([25, nmc]):
        mutrate, _, _, _ = transfer_predictions(features, nmc=nmc, scaled_popsize=scaled_popsize, genomesize=genomesize, seed=seed)
        mutrate = mutrate[(np.isfinite(mutrate)) & (frequency > 0)]

    return (mutrate, time, fitness, frequency)