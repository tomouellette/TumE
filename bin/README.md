# Inferring cancer evolution from single tumour biopsies using synthetic supervised learning 

## /bin/ overview

This folder contains general scripts for: 
 * simulating synthetic tumour sequencing data with empirically relevant parameters using CanEvolve.jl
     * `synthetic.jl`
 * training and evaluating 1D convolutional neural networks using synthetic tumour sequencing data
     * `random_search.py`
     * `nets.py`
 * evaluating performance of previously developed mixture models and single statistics on synthetic data
     * `metrics.R`
 * nearest neighbour search between simulated data and patient sequencing data
     * `specification_features.py`
 * transfer learning using an alternative cancer evolution simulator
     * `/temulator/` contains a python wrapper and additional scripts for [TEMULATOR](https://t-heide.github.io/TEMULATOR/) simulator
     * `transfer_learning.py`
     * `transfer_benchmark.py`
 * general helper scripts
     * `utils.py`
     * `parsing.py`