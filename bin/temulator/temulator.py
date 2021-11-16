# Hardcode the environment as need custom install for rpy2 on cluster
import os
os.environ['R_HOME'] = '/.mounts/labs/awadallalab/private/touellette/sources/R-4.0.3/'

import rpy2
from rpy2.robjects.packages import importr
from rpy2 import robjects
import numpy as np
import pandas as pd
from termcolor import colored

"""
A python wrapper class for the R-based cancer evolution simulator TEMULATOR (https://github.com/T-Heide/TEMULATOR)
"""
def frequencyThresholds(depth, alt_reads = 2):
    """
    Returns the upper and lower VAF cutoffs based on 2 or 3 binomial standard deviations
    """
    f_min = (alt_reads/depth) + ( ( 2.0*np.sqrt(alt_reads*(1-(alt_reads/depth))) ) / depth)
    f_max = 0.5 - ( ( 3.0*np.sqrt((0.5*depth)*(1-0.5)) ) / depth)
    return f_min, f_max

class Temulator:
    """
    A Simulator class for generating cancer evolution VAF distributions using TEMULATOR (Heide et al.)
    """
    def __init__(self, silent = False) -> None:        
        super(Temulator, self).__init__()
        if silent == False:
            print(colored('Simulator wrapper for TEMULATOR.', 'red'))
            print(colored('Source:', 'blue') + ' https://github.com/T-Heide/TEMULATOR')
            print('')
            print(colored('Parameters for this simulator:', 'magenta'))
            print(' - birthrates (list)')
            print(' - deathrates (list)')
            print(' - mutation_rates (list)')
            print(' - clone_start_times (list)')
            print(' - fathers (list)')
            print(' - simulation_end_time (float/int)')
            print(' - number_clonal_mutations (int)')
            print(' - depth (int)')
            print(' - depth_model (int)')
            print(' - min_vaf (float)')
            print(' - purity (float)')
            print(' - seed (NoneType/int)')
            print('')
    def run(self, parameters = None, seed = None) -> dict:        
        """
        Runs the TEMULATOR simulation and stores/returns sequencing, clone, and parameter outputs
        
        parameters is a dictionary containing:
        
        :param birthrates (list) Birth rates of clones. First index is background/founder population, subsequent indices are subclones (default: [1, 1])
        :param deathrates (list) Death rates of clones. First index is background/founder population, subsequent indices are subclones (default: [0.2, 0.2])
        :param mutation_rates (list) Mutation rates of clones. First index is background/founder population, subsequent indices are subclones (default: [40, 40])
        :param clone_start_times (list) Clone birth times (in N). First index is background/founder population, subsequent indices are subclones (default: [0, 256])
        :param fathers (list) The progenitor of a given clone (default: [0, 0])
        :param simulation_end_time (int) The final tumour population size (default: 1000)
        :param number_clonal_mutations (int) The number of clonal mutations on tumor initiation (default: 500)
        :param depth (int) The mean sequencing depth in the tumor population (default: 120)
        :param depth_model (int) The sequencing noise model (default: 1)
        :param min_vaf (float) The minimum detectable allele frequency in biopsy (default: 0.02)
        :param purity (float/int) The purity (proportion of cells that are tumour origin) in biopsy (default: 1)

        See https://github.com/T-Heide/TEMULATOR for further details on each individual parameter
        """
        if parameters == None:
            parameters = {
                'birthrates': [1, 1.4],
                'deathrates': [0.2, 0.2],
                'mutation_rates': [50, 50],
                'clone_start_times': [0, 256],
                'fathers': [0, 0],
                'simulation_end_time': 1000000,
                'number_clonal_mutations': 500,
                'depth': 100,
                'depth_model': 2,
                'min_vaf': 0.05,
                'purity': 1
            }
        #
        if seed == None:
            seed = np.random.randint(2e9)
        #
        self.seed = seed
        #    
        # Run TEMULATOR simulation
        temulator = importr('TEMULATOR')        
        sim = temulator.simulateTumour(
            birthrates = robjects.IntVector(parameters['birthrates']),
            deathrates = robjects.IntVector(parameters['deathrates']),
            mutation_rates = robjects.IntVector(parameters['mutation_rates']),
            clone_start_times = robjects.IntVector(parameters['clone_start_times']),
            fathers = robjects.IntVector(parameters['fathers']),
            simulation_end_time = parameters['simulation_end_time'],
            seed = seed,
            number_clonal_mutations = parameters['number_clonal_mutations'],
            purity = parameters['purity'],
            min_vaf = parameters['min_vaf'],
            depth = parameters['depth'],
            depth_model = parameters['depth_model']
        )
        #
        # Processing functions from TEMULATOR source code
        robjects.r('''
                   # >> Code sourced from TEMULATOR (https://github.com/T-Heide/TEMULATOR)
                   get_sequencing_data = function(x, idx=1, ...) {
                       library(dplyr)
                       assign_mutation_label = function(d) {
                           stopifnot(is.data.frame(d))
                           stopifnot(c("clone","id") %in% colnames(d))
                           f_group = base::strtoi(paste0("0x", gsub("X.*$", "", d$id)))
                           f_group = f_group - min(f_group)
                           c_group = d$clone
                           c_group = ifelse(c_group == 0, 1, c_group - min(c_group[c_group != 0]) + 2)
                           c_group[f_group == 0] = 0
                           label = as.character(c_group)
                        }
                        
                        if ("mutation_subsets" %in% names(x)) {
                            n_subsets = length(x$mutation_subsets[[1]])
                        } else {
                            n_subsets = 0
                        }
                        
                        if (!is.data.frame(x$sequencing_parameters)) { # old structure, one sequencing dataset
                            x$mutation_data = list(x$mutation_data)      # convert to new structure
                            if ("mutation_subsets" %in% names(x)) {
                                for (i in seq_along(x$mutation_subsets)) {
                                    x$mutation_subsets = list(x$mutation_subsets)
                                }
                            }
                        }
                        
                        n_datasets = length(x$mutation_data)
                        max_idx = (n_subsets + 1) * n_datasets
                        if (!is.numeric(idx)) stop("Index not numeric.")
                        if (!length(idx) == 1) stop("Index not of length 0.")
                        if (!length(idx) <= max_idx & idx > 0) stop("Index out of range.")
                        idx_a = (idx-1) %/% (n_subsets+1) + 1
                        idx_b = (idx-1) %% (n_subsets+1)
                        if (idx_b == 0) {
                            mdata = x$mutation_data[[idx_a]]
                        } else {
                            mdata = x$mutation_subsets[[idx_a]][[idx_b]]
                        }
                        mdata %>% 
                        mutate(vaf=alt/depth) %>% 
                        mutate(label=assign_mutation_label(., ...)) %>% 
                        select(alt, depth, vaf, id, label)
                    }
                    '''
        )
        #        
        # Get sequencing data
        r_get_sequencing_data = robjects.r['get_sequencing_data']
        df = pd.DataFrame(np.transpose(r_get_sequencing_data(sim)))
        df.columns = ['alt', 'depth', 'vaf', 'id', 'label']
        df = df.astype({'alt':'int', 'depth':'int', 'vaf':'float', 'id':'string', 'label':'int'})
        self.sequencing_data = df
        #
        # Get clone frequencies
        self.clone_frequencies = np.array(sim[3]) / sum(sim[3])
        #
        # Store simulation parameters
        self.parameters = parameters
        #
        # Return simulation data
        return {
            'sequencing_data': self.sequencing_data,
            'clone_frequencies': self.clone_frequencies,
            'parameters': self.parameters
        }        
        #
    def process(self, k = [64,128], bounds = [0.02, 0.5]) -> np.ndarray:
        """
        Generates a VAF distribution from mutations in the population sample

        Note:
            - Preprocess data like Caravagna et al. e.g. only VAFS >5%
        """
        # Convert VAF distribution
        vaf = np.array(self.sequencing_data['vaf'])
        f_min, f_max = frequencyThresholds(depth = self.parameters['depth'])
        vaf = vaf[vaf > f_min]
        self.features = np.hstack([np.histogram(vaf, bins = i, range = bounds)[0] for i in k])
        # Get labels
        mutrate = self.parameters['mutation_rates'][0]
        birth_times = np.array(self.parameters['clone_start_times'])[1:]
        fitness = (np.array(self.parameters['birthrates'])[1:] - np.array(self.parameters['deathrates'])[1:]) / (self.parameters['birthrates'][0] - self.parameters['deathrates'][0])
        # Save training data        
        return self.features, np.array([mutrate, birth_times, fitness, self.clone_frequencies[1:]], dtype=object)