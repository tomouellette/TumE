# ----------------------------------------------------------------------------------
# synthetic.jl: Various helper functions for building synthetic datasets for analysis
# ----------------------------------------------------------------------------------

"""

This julia script runs all simulation/synthetic data generation used Ouellette and Awadalla (2020).
The analyses using non-normalized VAF distributions (D - G). Cannot guarantee stability of (A - C) functions.

-------------------------
||| Table of contents |||
-------------------------

Exploratory (normalized VAF distributions):
    A. Generation of simulation/synthetic datasets under a realistic parameter setting regime for deep learning model training
    B. Simulated tumours for comparing existing methods (at different depths/dispersion) versus a deep learning approach
    C. Impact of varying birth and death rates on inferences

Described in study (non-normalized VAF distributions):
    D. Non-normalized VAF vectors for inference
    E. Simulated tumours for comparing existing methods (at different depths/dispersion) versus a deep learning approach using non-normalized VAF distributions
    F. Impact of varying birth and death rates on inferences using non-normalized VAFs

"""

# Load modules and tumour evolution code
include("/.mounts/labs/awadallalab/private/touellette/projects/TumE/src/CanEvolve.jl")
using PyCall
using Random
np = pyimport("numpy")

# Arguments
WHICH_SYNTHETIC = ARGS[1]
OUTPUT_DIR = ARGS[2] # Requires / at end of directory call


#A. Generation of simulation/synthetic datasets under a realistic parameter setting regime for deep learning model training
if WHICH_SYNTHETIC == "A"

    function synthetic_A(b, d, nsims, lower_cutoff = 0.09, upper_cutoff = 0.41)
        store = []
        for i in 1:nsims
            try
                p1, n1 = Main.CanEvolve.autoSimulation(b, d, Nfinal = 1000, noise = "betabinomial", nsubclone_min = 1, nsubclone_max = 1, lower_cutoff = lower_cutoff, upper_cutoff = upper_cutoff)
                p2, n2 = Main.CanEvolve.autoSimulation(b, d, Nfinal = 1000, noise = "betabinomial", nsubclone_min = 2, nsubclone_max = 2, lower_cutoff = lower_cutoff, upper_cutoff = upper_cutoff)
                p1f, p1l = Main.CanEvolve.engineer(p1, k = [64, 128, 256], lower_cutoff = lower_cutoff, upper_cutoff = upper_cutoff)
                p2f, p2l = Main.CanEvolve.engineer(p2, k = [64, 128, 256], lower_cutoff = lower_cutoff, upper_cutoff = upper_cutoff)
                n1f, n1l = Main.CanEvolve.engineer(n1, k = [64, 128, 256], lower_cutoff = lower_cutoff, upper_cutoff = upper_cutoff)
                n2f, n2l = Main.CanEvolve.engineer(n2, k = [64, 128, 256], lower_cutoff = lower_cutoff, upper_cutoff = upper_cutoff)
                push!(store, [p1f, p1l])
                push!(store, [p2f, p2l])
                push!(store, [n1f, n1l])
                push!(store, [n2f, n2l])
            catch e
                continue # Catch any errors that may occur on cluster runs so script doesn't crash
            end
        end
        return store
    end

    out = synthetic_A(log(2), 0.0, 5000)
    randid = randstring(10)
    np.save(OUTPUT_DIR * randid * ".npy", out)

end

# B. Simulated tumours for comparing existing methods (at different depths/dispersion) versus a deep learning approach
if WHICH_SYNTHETIC == "B"

    function synthetic_B(b, d, nsims, depth = 100, rho = 0.001, lower_cutoff = 0.09, upper_cutoff = 0.41)
        store = []
        sequence = []
        for i in 1:nsims
            try
                p1, n1 = Main.CanEvolve.autoSimulation_fixed(b, d, u = 100, depth = depth, Nfinal = 1000, noise = "betabinomial", rho = rho, nsubclone_min = 1, nsubclone_max = 1, lower_cutoff = lower_cutoff, upper_cutoff = upper_cutoff)
                p2, n2 = Main.CanEvolve.autoSimulation_fixed(b, d, u = 100, depth = depth, Nfinal = 1000, noise = "betabinomial", rho = rho, nsubclone_min = 2, nsubclone_max = 2, lower_cutoff = lower_cutoff, upper_cutoff = upper_cutoff)
                p1f, p1l = Main.CanEvolve.engineer(p1, k = [64, 128, 256], lower_cutoff = lower_cutoff, upper_cutoff = upper_cutoff)
                p2f, p2l = Main.CanEvolve.engineer(p2, k = [64, 128, 256], lower_cutoff = lower_cutoff, upper_cutoff = upper_cutoff)
                n1f, n1l = Main.CanEvolve.engineer(n1, k = [64, 128, 256], lower_cutoff = lower_cutoff, upper_cutoff = upper_cutoff)
                n2f, n2l = Main.CanEvolve.engineer(n2, k = [64, 128, 256], lower_cutoff = lower_cutoff, upper_cutoff = upper_cutoff)
                push!(store, [p1f, p1l])
                push!(store, [p2f, p2l])
                push!(store, [n1f, n1l])
                push!(store, [n2f, n2l])
                push!(sequence, [p1[2][1], p1[2][3]])
                push!(sequence, [p2[2][1], p2[2][3]])
                push!(sequence, [n1[2][1], n1[2][3]])
                push!(sequence, [n2[2][1], n2[2][3]])
            catch e
                continue # Catch any errors that may occur on cluster runs so script doesn't crash
            end
        end
        return store, sequence
    end

    for d in [50, 100, 150, 200, 250]
        for n in [0.0, 0.001, 0.003, 0.01, 0.03]
            store, sequence = synthetic_B(log(2), 0.0, 500, d, n)
            randid = randstring(10)
            np.save(OUTPUT_DIR * randid * "_" * string(d) * "_" * string(n) * "_features.npy", store)
            np.save(OUTPUT_DIR * randid * "_" * string(d) * "_" * string(n) * "_vafdepth.npy", sequence)
        end
    end

end

# C. Impact of varying birth and death rates on inferences
if WHICH_SYNTHETIC == "C"

    function synthetic_C(b, d, nsims, rho = 0.001, lower_cutoff = 0.09, upper_cutoff = 0.41)
        store = []
        for i in 1:nsims
            try
                p1, n1 = Main.CanEvolve.autoSimulation_fixed(b, d, u = 100, depth = 100, Nfinal = 1000, noise = "betabinomial", rho = rho, nsubclone_min = 1, nsubclone_max = 1, lower_cutoff = lower_cutoff, upper_cutoff = upper_cutoff)
                p2, n2 = Main.CanEvolve.autoSimulation_fixed(b, d, u = 100, depth = 100, Nfinal = 1000, noise = "betabinomial", rho = rho, nsubclone_min = 2, nsubclone_max = 2, lower_cutoff = lower_cutoff, upper_cutoff = upper_cutoff)
                p1f, p1l = Main.CanEvolve.engineer(p1, k = [64, 128, 256], lower_cutoff = lower_cutoff, upper_cutoff = upper_cutoff)
                p2f, p2l = Main.CanEvolve.engineer(p2, k = [64, 128, 256], lower_cutoff = lower_cutoff, upper_cutoff = upper_cutoff)
                n1f, n1l = Main.CanEvolve.engineer(n1, k = [64, 128, 256], lower_cutoff = lower_cutoff, upper_cutoff = upper_cutoff)
                n2f, n2l = Main.CanEvolve.engineer(n2, k = [64, 128, 256], lower_cutoff = lower_cutoff, upper_cutoff = upper_cutoff)
                push!(store, [p1f, p1l])
                push!(store, [p2f, p2l])
                push!(store, [n1f, n1l])
                push!(store, [n2f, n2l])
            catch e
                continue # Catch any errors that may occur on cluster runs so script doesn't crash
            end
        end
        return store
    end

    for b in [0.2, 0.4, 0.6, 0.8, 1.0, 2.0]
        for d in [0.0, 0.1, 0.2, 0.3, 0.4]
            if d >= b
                continue
            else
                store = synthetic_C(b, d, 500)
                randid = randstring(10)
                np.save(OUTPUT_DIR * randid * "_" * string(b) * "_" * string(d) * ".npy", store)
            end
        end
    end

end

# D. Examining utility of non-normalized VAF vectors for inference
if WHICH_SYNTHETIC == "D"

    function synthetic_D(b, d, nsims, lower_cutoff = 0.09, upper_cutoff = 0.41)
        store = []
        for i in 1:nsims
            try
                p1, n1 = Main.CanEvolve.autoSimulation(b, d, Nfinal = 1000, noise = "betabinomial", nsubclone_min = 1, nsubclone_max = 1, lower_cutoff = lower_cutoff, upper_cutoff = upper_cutoff)
                p2, n2 = Main.CanEvolve.autoSimulation(b, d, Nfinal = 1000, noise = "betabinomial", nsubclone_min = 2, nsubclone_max = 2, lower_cutoff = lower_cutoff, upper_cutoff = upper_cutoff)
                p1f, p1l = Main.CanEvolve.engineer_un(p1, k = [64, 128], lower_cutoff = lower_cutoff, upper_cutoff = upper_cutoff)
                p2f, p2l = Main.CanEvolve.engineer_un(p2, k = [64, 128], lower_cutoff = lower_cutoff, upper_cutoff = upper_cutoff)
                n1f, n1l = Main.CanEvolve.engineer_un(n1, k = [64, 128], lower_cutoff = lower_cutoff, upper_cutoff = upper_cutoff)
                n2f, n2l = Main.CanEvolve.engineer_un(n2, k = [64, 128], lower_cutoff = lower_cutoff, upper_cutoff = upper_cutoff)
                push!(store, [p1f, p1l])
                push!(store, [p2f, p2l])
                push!(store, [n1f, n1l])
                push!(store, [n2f, n2l])
            catch e
                continue # Catch any errors that may occur on cluster runs so script doesn't crash
            end
        end
        return store
    end

    out = synthetic_D(log(2), 0.0, 5000)
    randid = randstring(10)
    np.save(OUTPUT_DIR * randid * ".npy", out)

end

# E. Simulated tumours for comparing existing methods (at different depths/dispersion) versus a deep learning approach using non-normalized VAF distributions
if WHICH_SYNTHETIC == "E"

    function synthetic_E(b, d, nsims, depth = 100, rho = 0.001, lower_cutoff = 0.09, upper_cutoff = 0.41)
        store = []
        sequence = []
        for i in 1:nsims
            try
                p1, n1 = Main.CanEvolve.autoSimulation_fixed(b, d, u = 100, depth = depth, Nfinal = 1000, noise = "betabinomial", rho = rho, nsubclone_min = 1, nsubclone_max = 1, lower_cutoff = lower_cutoff, upper_cutoff = upper_cutoff, trim = false)
                p2, n2 = Main.CanEvolve.autoSimulation_fixed(b, d, u = 100, depth = depth, Nfinal = 1000, noise = "betabinomial", rho = rho, nsubclone_min = 2, nsubclone_max = 2, lower_cutoff = lower_cutoff, upper_cutoff = upper_cutoff, trim = false)
                p1f, p1l = Main.CanEvolve.engineer_un(p1, k = [64, 128], lower_cutoff = lower_cutoff, upper_cutoff = upper_cutoff)
                p2f, p2l = Main.CanEvolve.engineer_un(p2, k = [64, 128], lower_cutoff = lower_cutoff, upper_cutoff = upper_cutoff)
                n1f, n1l = Main.CanEvolve.engineer_un(n1, k = [64, 128], lower_cutoff = lower_cutoff, upper_cutoff = upper_cutoff)
                n2f, n2l = Main.CanEvolve.engineer_un(n2, k = [64, 128], lower_cutoff = lower_cutoff, upper_cutoff = upper_cutoff)
                push!(store, [p1f, p1l])
                push!(store, [p2f, p2l])
                push!(store, [n1f, n1l])
                push!(store, [n2f, n2l])
                push!(sequence, [p1[2][1], p1[2][3]])
                push!(sequence, [p2[2][1], p2[2][3]])
                push!(sequence, [n1[2][1], n1[2][3]])
                push!(sequence, [n2[2][1], n2[2][3]])
            catch e
                continue # Catch any errors that may occur on cluster runs so script doesn't crash
            end
        end
        return store, sequence
    end

    for d in [50, 75, 100,  125, 150, 200, 250]
        for n in [0.0, 0.001, 0.003, 0.01, 0.03]
            store, sequence = synthetic_E(log(2), 0.0, 100, d, n)
            randid = randstring(10)
            np.save(OUTPUT_DIR * randid * "_" * string(d) * "_" * string(n) * "_features.npy", store)
            np.save(OUTPUT_DIR * randid * "_" * string(d) * "_" * string(n) * "_vafdepth.npy", sequence)
        end
    end

end

# F. Impact of varying birth and death rates on inferences using non-normalized VAFs
if WHICH_SYNTHETIC == "F"

    function synthetic_F(b, d, nsims, rho = 0.001, lower_cutoff = 0.09, upper_cutoff = 0.41)
        store = []
        for i in 1:nsims
            try
                p1, n1 = Main.CanEvolve.autoSimulation_fixed(b, d, u = 100, depth = 100, Nfinal = 1000, noise = "betabinomial", rho = rho, nsubclone_min = 1, nsubclone_max = 1, lower_cutoff = lower_cutoff, upper_cutoff = upper_cutoff, trim = false)
                p2, n2 = Main.CanEvolve.autoSimulation_fixed(b, d, u = 100, depth = 100, Nfinal = 1000, noise = "betabinomial", rho = rho, nsubclone_min = 2, nsubclone_max = 2, lower_cutoff = lower_cutoff, upper_cutoff = upper_cutoff, trim = false)
                p1f, p1l = Main.CanEvolve.engineer_un(p1, k = [64, 128], lower_cutoff = lower_cutoff, upper_cutoff = upper_cutoff)
                p2f, p2l = Main.CanEvolve.engineer_un(p2, k = [64, 128], lower_cutoff = lower_cutoff, upper_cutoff = upper_cutoff)
                n1f, n1l = Main.CanEvolve.engineer_un(n1, k = [64, 128], lower_cutoff = lower_cutoff, upper_cutoff = upper_cutoff)
                n2f, n2l = Main.CanEvolve.engineer_un(n2, k = [64, 128], lower_cutoff = lower_cutoff, upper_cutoff = upper_cutoff)
                push!(store, [p1f, p1l])
                push!(store, [p2f, p2l])
                push!(store, [n1f, n1l])
                push!(store, [n2f, n2l])
            catch e
                continue # Catch any errors that may occur on cluster runs so script doesn't crash
            end
        end
        return store
    end

    for b in [0.2, 0.4, 0.6, 0.8, 1.0, 2.0]
        for d in [0.0, 0.1, 0.2, 0.3, 0.4]
            #[[0.2, 0.3], [0.2, 0.6], [0.2, 0.9], [0.2, 1.2], [0.2, 1.6], [0.4. 0.6], [0.4, 0.9], [0.4, 1.2], [0.4, 1.6]]
            if d >= b
                continue
            else
                store = synthetic_F(b, d, 100)
                randid = randstring(10)
                np.save(OUTPUT_DIR * randid * "_" * string(b) * "_" * string(d) * ".npy", store)
            end
        end
    end

end