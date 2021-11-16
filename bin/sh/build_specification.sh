#!/bin/bash

# Build cluster scripts for computing nearest neighbour search with empirical samples

main='/.mounts/labs/awadallalab/private/touellette'
path="${main}/projects/TumE/synthetic/D/"
empirical="${main}/projects/TumE/data/adjusted_all.csv"
outputdir="${main}/projects/TumE/data/"

echo '#!/bin/bash' > "run_specification.sh"
echo '#$ -P awadallalab' >> "run_specification.sh"
echo '#$ -cwd' >> "run_specification.sh"
echo '#$ -l h_vmem=4G' >> "run_specification.sh"
echo '#$ -l h_rt=48:00:00' >> "run_specification.sh"
echo '#$ -pe smp 20' >> "run_specification.sh"
echo '#$ -V' >> "run_specification.sh"		
echo 'module load python/3.6' >> "run_specification.sh"
echo "source ${main}/python/bin/activate" >> "run_specification.sh"
printf "python3 ${main}/projects/TumE/bin/specification_features.py -p ${path} -e ${empirical} -od ${outputdir}" >> "run_specification.sh"