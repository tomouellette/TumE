#!/bin/env Rscript

# --------------------------------------------------------------------------
# Checking viable fitnesses/emergence times for one subclone using TEMULATOR
# --------------------------------------------------------------------------

source('../temulator/temulator_main.R')
library(data.table)

args = commandArgs(trailingOnly = TRUE)
output_path = args[1]

frequencies = c()
birthrates = c()
times = c()
replicates = 10
for (i in seq(1, 10, 0.25)) {
  print(paste0('On to birth rate ', i))
  for (j in seq(1, 13, 0.5)) {
    print(j)
    for (q in 1:replicates) {
      out = run_1subclone_temulator(b = i, u = 20, t = 2^j, dp = 100, nclonal = 500, Nfinal = 1e4, engineer = FALSE)
      frequencies = append(frequencies, out[[2]][1])
      birthrates = append(birthrates, i)
      times = append(times, j)
    }
  }
}

out = data.table(f = frequencies, b = birthrates, t = times)
write.table(out, file = paste0(output_path, 'temulator_viable_1subclone.tsv'), quote = FALSE, sep = '\t', row.names = FALSE)