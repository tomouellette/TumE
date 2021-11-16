#!/bin/env Rscript
library(data.table)
library(dplyr)
library(mobster)
library(neutralitytestr)
library(doParallel)
library(foreach)
library(reticulate)

# -------------------------------------------------------------------------------------------
# A command-line script to run R-based evolution statistics on simulated data in numpy format
# -------------------------------------------------------------------------------------------

# command-line argument 1: Index for running jobs, subsets directory by [i, i+1] jobs; maximum i = N jobs
# command-line argument 2: N jobs to split directory into e.g 500
args = commandArgs(trailingOnly = TRUE)

# Example:
# Given a directory of 1000 samples 
# >> Rscript 10 500
# This would split the directory 500 times with each analysis consisting of 2 (1000/500) samples each
# And 10 indicates that the 2 samples consist of the [10, 11] indices of this split

# Hardcoded paths
VIRTUAL_ENV = '/.mounts/labs/awadallalab/private/touellette/python/'
BENCHMARK_DIR = '/.mounts/labs/awadallalab/private/touellette/projects/TumE/synthetic/E_benchmark/'
OUTPUT_DIR = '/.mounts/labs/awadallalab/private/touellette/projects/TumE/analysis/E/'

runMobster <- function(vaf, reads, depth, slow = FALSE) {
    #' Run MOBSTER (Caravagna et al. 2020) on synthetic or real tumour sequencing data
    #'
    #' @param vaf The variant allele frequency for each mutation
    #' @param reads The number of reads covering each mutation
    #' @param depth The sequencing depth (alt + reference reads) covering each mutation
    #' @return A vector containing the number of subclones, tail identification, and clonal and subclonal frequencies (up to 2 subclones)
    #'
    tryCatch(
    	expr = {
            # Noticed odd behaviour of MOBSTER when including variants below 5% VAF -- therefore we threshold data >5% VAF
            reads = reads[(vaf < 1) & (vaf > 0.05)]
            depth = depth[(vaf < 1) & (vaf > 0.05)]
            vaf = vaf[(vaf < 1) & (vaf > 0.05)]
            data = data.table(t_alt_count = reads, DP = depth, VAF = vaf)
            if (slow == TRUE) {
                # Far too slow for >millions of samples
                fit = mobster_fit(data, parallel = FALSE)
            } else {
            	fit = mobster_fit(data, K = 1:3, samples = 2, init = "peaks", tail = c(TRUE, FALSE), epsilon = 1e-6, maxIter = 100, fit.type = "MM", seed = 12345, model.selection = "reICL", trace = FALSE, parallel = FALSE, pi_cutoff = 0.02, N_cutoff = 10)
            }

            # Get C1 subclone frequency
            index = which(paste(fit$best$Clusters$type, fit$best$Clusters$cluster) == 'Mean C1')
            if (length(index) != 0) {
            	c1freq = fit$best$Clusters$fit.value[which(paste(fit$best$Clusters$type, fit$best$Clusters$cluster) == 'Mean C1')]
            } else {
            	c1freq = 0
            }

            # Get C2 subclone frequency
            index = which(paste(fit$best$Clusters$type, fit$best$Clusters$cluster) == 'Mean C2')
            if (length(index) != 0) {
            	c2freq = fit$best$Clusters$fit.value[which(paste(fit$best$Clusters$type, fit$best$Clusters$cluster) == 'Mean C2')]
            } else {
            	c2freq = 0
            }

            # Get C3 subclone frequency
            index2 = as.numeric(which(paste(fit$best$Clusters$type, fit$best$Clusters$cluster) == 'Mean C3'))
            if (length(index2) != 0) {
            	c3freq = fit$best$Clusters$fit.value[which(paste(fit$best$Clusters$type, fit$best$Clusters$cluster) == 'Mean C3')]
            } else {
            	c3freq = 0
            }

            # Count clones
            tail = fit$best$fit.tail
            nsubclonal = 0

            # MOBSTER A definition of subclone
            # - If C2 subclone and C1 clonal peak present, but no tail, then no subclone
            # - If C2 subclone and C1 clonal peak present, with tail, then subclone
            # - If C2 subclone, C3 subclone, and C1 clonal peak present, but no tail, then 1 subclone
            # - If C2 subclone, C3 subclone, and C1 clonal peak present, with tail, then subclone
            if ((c2freq > 0) & (c3freq == 0) & (tail == TRUE)) {
            	nsubclonal = 1
            }
            if ((c2freq > 0) & (c3freq > 0) & (tail == FALSE)) {
            	nsubclonal = 1
            }
            if ((c2freq > 0) & (c3freq > 0)) {
            	nsubclonal = 2
            }

            # Evaluate
            return(c(nsubclonal, tail, c1freq, c2freq, c3freq))
        },
        error = function(e){
        	return(c(0, 0, 0, 0, 0))
    })

}

TajWu <- function(data, coverage = 100, upperbound = 0.5, lowerbound = 0, alt_reads = 2) {
    #' Compute population genetic statistics: Tajima's D and Fay and Wu's H
    #'
    #' @param data The variant allele frequency for each mutation
    #' @param coverage The mean sequencing depth for the given synthetic or real tumour sequencing data
    #' @param upperbound The upper bound of variant allele frequencies for computing statistics
    #' @param lowerbound The lower bound of variant allele frequencies for computing statistics
    #' @param alt_reads The minimum number of alternate reads to call a variant
    #' @return A vector with the computed Tajima's D and Fay and Wu's H
    #'
    if (upperbound == 0.5 & lowerbound == 0) {

    	upperbound = 0.5 - ( (3 * sqrt( (0.5 * coverage) * (1 - 0.5) ) ) / coverage )
    	lowerbound = (alt_reads / coverage) + ( 3 * sqrt( ((alt_reads / coverage) * coverage) * (1 - (alt_reads / coverage) ) ) / coverage )
    
    }

    data = data[data < upperbound & data > lowerbound]
    data = data * 2 # Adjust to 0 - 1 frequency

    # Use coverage as a proxy for N of each mutant in the population
    n <- coverage

    # Watterson's estimator
    S <- length(data) # Number of segregating sites
    an <- sum(1 / seq(1, n-1))
    theta_w <- S / an

    # Tajima's pi
    d <- sum((data * (1 - data))) # Estimated number of pairwise differences
    d <- d * n^2
    theta_pi <- d / choose(n, 2)

    # Tajima's a1, a2, b1, b2, c1, c2, e1, e2
    a1 <- sum(1 / seq(1, n - 1))
    a2 <- sum(1 / (seq(1, n - 1)^2))
    b1 <- (n + 1) / (3 * (n - 1))
    b2 <- (2 * (n^2 + n+3)) / (9*n * (n-1))
    c1 <- (b1) - (1 / a1)
    c2 <- (b2) - ((n + 2) / (a1 * n)) + (a2/a1^2)
    e1 <- (c1 / a1)
    e2 <- c2 / (a1^2 + a2)

    # Tajima's D
    seD <- sqrt((e1 * S) + ((e2 * S)*(S-1)))
    TajimasD <- (theta_pi - theta_w) / seD

    # Fay and Wu's H
    theta_l <- sum(data) * (n / (n - 1))
    bn <- sum(1 / (seq(1, n-1)^2))
    bn1 <- sum(1 / (seq(1, n)^2))
    theta2 <- (S * (S - 1)) / (an^2 + bn)
    seH <- sqrt((((n - 2) / (6*(n - 1))) * theta_w) + (theta2 * (((18 * n^2 * (3 * n + 2) * bn1) - (88 * n^3 + 9 * n^2 - 13*n + 6))/(9 * n * (n - 1)^2))))
    WuH <- 2 * ((theta_pi - theta_l) / seH)

    return(c(TajimasD, WuH))

}

computeMetrics <- function(data, identifier, alt_reads = 2) {
    #' Computes MOBSTER fits (Caravagna et al. 2020), Neutraly theory fits (Williams et al. 2016/2018), and classic population genetic statistics (Tajima, Fay and Wu's)
    #'
    #' @param data A vector containing [variant allele frequencies, reads, depth] for each mutation
    #' @param identifier A filename identifier containing depth and sequencing dispersion info
    #' @param alt_reads The minimum number of alternate reads to call a variant
    #' @return A single row data table with all computed metrics
    #'	
    VAF = data[1,]
    reads = data[2,]
    depth = data[3,]

    dp = as.numeric(strsplit(identifier, '_')[[1]][2])
    rho = as.numeric(strsplit(identifier, '_')[[1]][3])
    coverage = mean(depth)

    # Compute 'general' population genetic summary statistics
    TajD = TajWu(data = VAF, coverage = coverage)[1]
    WuH = TajWu(data = VAF, coverage = coverage)[2]

    # Get subclonal decomposition information
    mob = runMobster(VAF, reads, depth)

    # Cancer evolution statistics: Area, R-squared, Kolmogorov distance, Distance (Williams et al. 2016/2018)
    nct = function(VAF, coverage, rho) { tryCatch(
    	expr = {
    	    # Use Williams et al. 2016 integration boundary 0.12,0.24
    	    neutrality = neutralitytest(VAF, read_depth = mean(depth), cellularity = 1, rho = rho, ploidy = 2)
    	    neutralArea  = neutrality$area[[1]]
    	    neutralRsqr = neutrality$rsq[[1]]
    	    neutralKold  = neutrality$Dk[[1]]
    	    neutralDist = neutrality$meanDist[[1]]
    	    return(c(neutralArea, neutralRsqr, neutralKold, neutralDist))
    	},
    	error = function(e){ # Run this error function if depth, rho combination lead to no frequency interval with default settings
    		return(c(-1, -1, -1, -1))
    })}

    nct = nct(VAF, coverage, rho)
    neutralArea  = nct[1]
    neutralRsqr = nct[2]
    neutralKold  = nct[3]
    neutralDist = nct[4]

    dt = data.table(identifier = identifier, depth = dp, rho = rho, TajD = TajD, WuH = WuH, neutralArea = neutralArea, neutralRsqr = neutralRsqr, neutralKold = neutralKold, neutralDist = neutralDist, mobsterNumberSubclones = mob[1], mobsterTail = mob[2], mobsterSubcloneFreqC1 = mob[3], mobsterSubcloneFreqC2 = mob[4], mobsterSubcloneFreqC3 = mob[5])

    return (dt)
}

# ==========
# >>>>> MAIN
# ==========

# Index directory for each job and subset based on interval number
identifiers = list.files(BENCHMARK_DIR)
identifiers = identifiers[identifiers %like% 'vafdepth.npy']
nfiles = length(identifiers)
indices = seq(1, nfiles, round(nfiles/as.numeric(args[2]), 0))
indices[(as.numeric(args[2]) + 1)] = nfiles
j = as.numeric(args[1])
subset = indices[j:(j+1)]
subset[1] = ifelse(j == 1, 1, subset[1]+1)
indices = seq(subset[1], subset[2], 1)
identifiers = identifiers[indices]

# Parallelize
setwd(BENCHMARK_DIR)
cl <- makeCluster(10)
registerDoParallel(cl)
store = foreach(i=identifiers, .combine = rbind, .packages = c("reticulate", "mobster", "data.table", "dplyr", "neutralitytestr")) %dopar% {
    tryCatch(
    	expr = {
            use_virtualenv(virtualenv = VIRTUAL_ENV)
            np = import("numpy")
            computeMetrics(data = np$load(i), identifier = i, alt_reads = 2)
        },
        error = function(e){
            data.table(identifier = i, depth = NA, rho = NA, TajD = NA, WuH = NA, neutralArea = NA, neutralRsqr = NA, neutralKold = NA, neutralDist = NA, mobsterNumberSubclones = NA, mobsterTail = NA, mobsterSubcloneFreqC1 = NA, mobsterSubcloneFreqC2 = NA, mobsterSubcloneFreqC3 = NA)
        })
}
stopCluster(cl)

# Save output
write.table(store, file = paste0(OUTPUT_DIR, "benchmark_", as.character(j), ".tsv"), quote = FALSE, row.names = FALSE, sep = '\t')

# -------- Appendix --------

# Non-parallel code
# =================
# store = data.table()
# use_virtualenv(virtualenv = VIRTUAL_ENV)
# np = import('numpy')
# for (i in identifiers) {
# 	data = np$load(paste0(BENCHMARK_DIR, i))
# 	dt = computeMetrics(data, i)
# 	store = rbind(store, dt)
# 	data = 0
# 	dt = 0
# }