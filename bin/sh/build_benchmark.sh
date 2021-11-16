#!/bin/bash

# Build cluster scripts for evaluating top 10 deep learning models on an independent validation across depth and rho

main='/.mounts/labs/awadallalab/private/touellette'
modeldir='/.mounts/labs/awadallalab/private/touellette/projects/TumE/analysis/models/'
datadir='/.mounts/labs/awadallalab/private/touellette/projects/TumE/synthetic/E/'
outputdir='/.mounts/labs/awadallalab/private/touellette/projects/TumE/analysis/E_TumE/'
for i in $(ls $modeldir); do
		echo '#!/bin/bash' > "run_benchmark_E_${i}.sh"
		echo '#$ -P awadallalab' >> "run_benchmark_E_${i}.sh"
		echo '#$ -cwd' >> "run_benchmark_E_${i}.sh"
		echo '#$ -l h_vmem=96G' >> "run_benchmark_E_${i}.sh"
		echo '#$ -l h_rt=72:00:00' >> "run_benchmark_E_${i}.sh"
		echo '#$ -V' >> "run_benchmark_E_${i}.sh"
		echo 'module load python/3.6' >> "run_benchmark_E_${i}.sh"
		echo "source ${main}/python/bin/activate" >> "run_benchmark_E_${i}.sh"
		printf "python3 ${main}/projects/TumE/bin/benchmark.py -d ${modeldir} -m ${i} -dt ${datadir} -od ${outputdir}" >> "run_benchmark_E_${i}.sh";
done

main='/.mounts/labs/awadallalab/private/touellette'
modeldir='/.mounts/labs/awadallalab/private/touellette/projects/TumE/analysis/models/'
datadir='/.mounts/labs/awadallalab/private/touellette/projects/TumE/synthetic/F/'
outputdir='/.mounts/labs/awadallalab/private/touellette/projects/TumE/analysis/F_TumE/'
for i in $(ls $modeldir); do
		echo '#!/bin/bash' > "run_benchmark_F_${i}.sh"
		echo '#$ -P awadallalab' >> "run_benchmark_F_${i}.sh"
		echo '#$ -cwd' >> "run_benchmark_F_${i}.sh"
		echo '#$ -l h_vmem=120G' >> "run_benchmark_F_${i}.sh"
		echo '#$ -l h_rt=72:00:00' >> "run_benchmark_F_${i}.sh"
		echo '#$ -V' >> "run_benchmark_F_${i}.sh"
		echo 'module load python/3.6' >> "run_benchmark_F_${i}.sh"
		echo "source ${main}/python/bin/activate" >> "run_benchmark_F_${i}.sh"
		printf "python3 ${main}/projects/TumE/bin/benchmark.py -d ${modeldir} -m ${i} -dt ${datadir} -od ${outputdir}" >> "run_benchmark_F_${i}.sh";
done