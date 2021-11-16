#!/bin/env Rscript
library(dplyr)
library(data.table)
library(TEMULATOR)
library(reticulate)
np = import('numpy')

# -------------------------------------------------------------------------------------------------
# Transfer learning using an alternate simulation framework >> https://github.com/T-Heide/TEMULATOR
# -------------------------------------------------------------------------------------------------

# Hardcoded paths
VENV = '/.mounts/labs/awadallalab/private/touellette/python/'
use_virtualenv(VENV)

# TEMULATOR source function
get_sequencing_data = function(x, idx=1, ...) {
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

uniformdensity = function(vaf, range_ = c(0.02, 0.5), k = 100, depth = 50, alt_reads = 2, cutoff = 1) {
  if (cutoff == 1) {
    min_cutoff = function(depth, alt_reads = 2) alt_reads/depth + runif(1, 3, n = 1) * (sqrt(alt_reads*(1-alt_reads/depth))/depth)
    f_min = min_cutoff(depth, alt_reads)
    h = np$histogram(vaf[((range_[2] > vaf) & (vaf > f_min))], range = np$array(range_), bins = np$int(k), density = TRUE)
    nd = h[1]
  } else {
    h = np$histogram(vaf[((range_[2] > vaf) & (vaf > 0.02))], range = np$array(range_), bins = np$int(k), density = TRUE)
    nd = h[1]
  }
  return(nd)
}

vaf2feature = function(vaf, depth, k = c(64, 128), alt_reads = 2) {
  depth = np$int(np$round(np$mean(depth)))
  features = c()
  for (bin_number in k) {
    ud1 = uniformdensity(unlist(vaf), k = bin_number, depth = depth, alt_reads = alt_reads, range_ = c(0.02, 0.5))
    features = append(features, ud1)
  }
  return(features)
}

run_1subclone_temulator = function(b, u, t, dp, nclonal, Nfinal = 10000, engineer = TRUE) {
  #' Runs a TEMULATOR simulation for evolutionary inference
  #'  
  #' @param b birth rate for subclone that dictates fitness advantage over background population with b = 1
  #' @param u mutation rate in per genome per division
  #' @param t time subclone emerges in number of reactions (i.e. population size)
  #' @param dp mean sequencing depth
  #' @param engineer returns vaf distribution in binned format for training deep learning models else returns all vafs
  
  min_cutoff = function(depth, alt_reads = 2) alt_reads/depth + runif(1, 3, n = 1) * (sqrt(alt_reads*(1-alt_reads/depth))/depth)
  
  sim = simulateTumour(
    birthrates = c(1, b),
    deathrates = c(0.2, 0.2),
    mutation_rates = c(u, u),
    clone_start_times = c(0, t),
    fathers = c(0, 0),
    simulation_end_time = Nfinal,
    seed = round(runif(1, 10000000, n = 1),0),
    number_clonal_mutations = nclonal,
    purity = 1,
    min_vaf = min_cutoff(dp),
    depth = dp,
    depth_model = 1,
    verbose = FALSE,
    subset_fractions = numeric()
  )
  
  # Get parameters
  frequency = as.vector((sim$cell_numbers/sum(sim$cell_numbers)/2)[2])
  clonal_mutations = as.vector(sim$sequencing_parameters[1])
  minimum_vaf = as.vector(sim$sequencing_parameters[3])
  depth = as.vector(sim$sequencing_parameters[4])
  fitness = as.vector((sim$clone_parameters$birthrates[2] - sim$clone_parameters$deathrates[1])/(sim$clone_parameters$birthrates[1]))
  time = as.vector(sim$clone_parameters$clone_start_times[2])
  mutrate = as.vector(sim$clone_parameters$mutation_rates[2])
  vaf = as.vector(get_sequencing_data(sim)$vaf)
  scmuts = as.vector(nrow(filter(sim$mutation_data, clone == 2)))
  
  if (engineer == TRUE) {
    vaf = np$hstack(vaf2feature(vaf, depth))
  }
  
  labels = c(frequency, scmuts, fitness, time, mutrate, clonal_mutations, depth, minimum_vaf)
  features = vaf
  
  return(list(features, labels, sim))
  
}

run_2subclone_temulator = function(b1, b2, u, t1, t2, dp, nclonal, Nfinal = 10000, engineer = TRUE) {
  #' Runs a TEMULATOR simulation for evolutionary inference
  #'  
  #' @param b birth rate for subclone that dictates fitness advantage over background population with b = 1
  #' @param u mutation rate in per genome per division
  #' @param t time subclone emerges in number of reactions (i.e. population size)
  #' @param dp mean sequencing depth
  #' @param engineer returns vaf distribution in binned format for training deep learning models else returns all vafs
  
  min_cutoff = function(depth, alt_reads = 2) alt_reads/depth + runif(1, 3, n = 1) * (sqrt(alt_reads*(1-alt_reads/depth))/depth)
  
  sim = simulateTumour(
    birthrates = c(1, b1, b2),
    deathrates = c(0.2, 0.2, 0.2),
    mutation_rates = c(u, u, u),
    clone_start_times = c(0, t1, t2),
    fathers = c(0, 0, 0),
    simulation_end_time = Nfinal,
    seed = round(runif(1, 10000000, n = 1),0),
    number_clonal_mutations = nclonal,
    purity = 1,
    min_vaf = min_cutoff(dp),
    depth = dp,
    depth_model = 1,
    verbose = FALSE,
    subset_fractions = numeric()
  )
  
  # Get parameters
  frequency = as.vector(sim$cell_numbers/(sum(sim$cell_numbers))/2)[2:3]
  clonal_mutations = as.vector(sim$sequencing_parameters[1])
  minimum_vaf = as.vector(sim$sequencing_parameters[3])
  depth = as.vector(sim$sequencing_parameters[4])
  fitness = as.vector((sim$clone_parameters$birthrates - sim$clone_parameters$deathrates)/(sim$clone_parameters$birthrates))[2:3]
  time = as.vector(sim$clone_parameters$clone_start_times)[2:3]
  mutrate = as.vector(sim$clone_parameters$mutation_rates[2])
  vaf = as.vector(get_sequencing_data(sim)$vaf)
  
  if (engineer == TRUE) {
    vaf = np$hstack(vaf2feature(vaf, depth))
  }
  
  labels = c(frequency[1], frequency[2], time[1], time[2], fitness[1], fitness[2], mutrate, clonal_mutations, depth, minimum_vaf)
  features = vaf
  
  return(list(features, labels, sim))
  
}