#!/bin/bash

# Build cluster scripts for evaluating different evolutionary inference methods

main='/.mounts/labs/awadallalab/private/touellette'
for i in {1..500}; do
		echo '#!/bin/bash' > "run_metrics_${i}.sh"
		echo '#$ -P awadallalab' >> "run_metrics_${i}.sh"
		echo '#$ -cwd' >> "run_metrics_${i}.sh"
		echo '#$ -l h_vmem=4G' >> "run_metrics_${i}.sh"
		echo '#$ -l h_rt=16:00:00' >> "run_metrics_${i}.sh"
		echo '#$ -pe smp 10' >> "run_metrics_${i}.sh"
		echo '#$ -V' >> "run_metrics_${i}.sh"		
		echo 'module load rstats' >> "run_metrics_${i}.sh"
		echo 'module load python/3.6' >> "run_metrics_${i}.sh"
		echo "source ${main}/python/bin/activate" >> "run_metrics_${i}.sh"
		printf "Rscript ${main}/projects/TumE/bin/metrics.R ${i} 500" >> "run_metrics_${i}.sh";
done