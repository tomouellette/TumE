#!/bin/env Rscript

# --------------------------------------------------------------------------
# Checking viable fitnesses/emergence times for one subclone using TEMULATOR
# --------------------------------------------------------------------------

source('../temulator/temulator_main.R')
library(data.table)

args = commandArgs(trailingOnly = TRUE)
output_path = args[1]

frequencies1 = c()
frequencies2 = c()
birthrates1 = c()
birthrates2 = c()
times1 = c()
times2 = c()
replicates = 5
for (i1 in seq(1, 10, 1)) {  
  for (i2 in seq(1, 10, 1)) {
    for (j1 in seq(1, 13, 1)) {
      for (j2 in seq(1, 13, 1)) {
        print(i1)
        print(i2)
        print(j1)
        print(j2)
        print('---')
        tryCatch(expr = {
          if ((j1 < j2) & (i1 < i2)) {
            for (q in 1:replicates) {
              out = run_2subclone_temulator(b1 = i1, b2 = i2, u = 20, t1 = 2^j1, t2 = 2^j2, dp = 100, nclonal = 500, Nfinal = 1e4, engineer = TRUE)
              frequencies1 = append(frequencies1, out[[2]][1])
              frequencies2 = append(frequencies2, out[[2]][2])
              birthrates1 = append(birthrates1, i1)
              birthrates2 = append(birthrates2, i2)
              times1 = append(times1, j1)
              times2 = append(times2, j2)
              out = 0
            }
          } else {
            next
          }
        }, error=function(e){
          print(e)
        }
        )
      }
    }
  }
}

out = data.table(f1 = frequencies1, f2 = frequencies2, b1 = birthrates1, b2 = birthrates2, t1 = times1, t2 = times2)
write.table(out, file = paste0(output_path, 'temulator_viable_2subclone.tsv'), quote = FALSE, sep = '\t', row.names = FALSE)