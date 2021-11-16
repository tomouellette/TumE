#!/bin/bash

main='/.mounts/labs/awadallalab/private/touellette'
viable="${main}/projects/TumE/temulator/temulator_viable_1subclone.tsv"
output="${main}/projects/TumE/temulator/test_set/"
for i in 75 100 125 150 200; do
		echo '#!/bin/bash' > "temulator_test_${i}.sh"
		echo '#$ -P awadallalab' >> "temulator_test_${i}.sh"
		echo '#$ -cwd' >> "temulator_test_${i}.sh"
		echo '#$ -l h_vmem=32G' >> "temulator_test_${i}.sh"
		echo '#$ -l h_rt=8:00:00' >> "temulator_test_${i}.sh"
		echo '#$ -V' >> "temulator_test_${i}.sh"		
		echo 'module load rstats' >> "temulator_test_${i}.sh"
		echo 'module load python/3.6' >> "temulator_test_${i}.sh"
		echo "source ${main}/python/bin/activate" >> "temulator_test_${i}.sh"
		# Need to update library path so that rpy2 can access libR.so from custom R install on cluster
		echo 'export LD_LIBRARY_PATH="/.mounts/labs/awadallalab/private/touellette/sources/R-4.0.3/lib:$LD_LIBRARY_PATH"' >> "temulator_test_${i}.sh"
		printf "python3 ${main}/projects/TumE/temulator/temulator_test_set.py --viable ${viable} --nsims 1000 --output ${output} --depth ${i}" >> "temulator_test_${i}.sh";
done

#export LD_LIBRARY_PATH="/.mounts/labs/awadallalab/private/touellette/sources/R-4.0.3/lib:$LD_LIBRARY_PATH"