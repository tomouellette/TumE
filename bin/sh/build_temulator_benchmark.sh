#!/bin/bash

main='/.mounts/labs/awadallalab/private/touellette'
modeldir="${main}/projects/TumE/analysis/transfer_models/"
datadir="${main}/projects/TumE/temulator/test_set/"
outputdir="${main}/projects/TumE/temulator/test_predictions/"
for i in `ls $modeldir`; do
		mod=$i
		echo '#!/bin/bash' > "temulator_benchmark_${i}.sh"
		echo '#$ -P awadallalab' >> "temulator_benchmark_${i}.sh"
		echo '#$ -cwd' >> "temulator_benchmark_${i}.sh"
		echo '#$ -l h_vmem=72G' >> "temulator_benchmark_${i}.sh"
		echo '#$ -l h_rt=8:00:00' >> "temulator_benchmark_${i}.sh"
		echo '#$ -V' >> "temulator_benchmark_${i}.sh"		
		echo 'module load python/3.6' >> "temulator_benchmark_${i}.sh"
		echo "source ${main}/python/bin/activate" >> "temulator_benchmark_${i}.sh"
		printf "python3 ${main}/projects/TumE/bin/transfer_benchmark.py  --dir ${modeldir} --mod ${mod} --data ${datadir} --outputdir ${outputdir}" >> "temulator_benchmark_${i}.sh";
done