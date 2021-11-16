#!/bin/bash

# Build cluster scripts for performing random hyperparameter search

main='/.mounts/labs/awadallalab/private/touellette'
datadir="${main}/projects/TumE/synthetic/D/"
tasks=('evolution' 'onesubclone' 'twosubclone')
inputdim=192
outdir="${main}/projects/TumE/analysis/D/"

for t in ${tasks[@]}; do
		echo '#!/bin/bash' > "run_random_search_${t}.sh"
		echo '#$ -P awadallalab' >> "run_random_search_${t}.sh"
		echo '#$ -cwd' >> "run_random_search_${t}.sh"
		echo '#$ -l h_vmem=96G' >> "run_random_search_${t}.sh"
		echo '#$ -l h_rt=72:00:00' >> "run_random_search_${t}.sh"
		echo '#$ -V' >> "run_random_search_${t}.sh"
		echo 'module load python/3.6' >> "run_random_search_${t}.sh"
		echo "source ${main}/python/bin/activate" >> "run_random_search_${t}.sh"
		printf "python3 ${main}/projects/TumE/bin/random_search.py --path ${datadir} --task ${t} --outputdir ${outdir}" >> "run_random_search_${t}.sh"
done