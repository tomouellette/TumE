#!/bin/bash

for size in 10 25 50 100 250 500 1000; do
	datadir="${main}/projects/TumE/temulator/train_set_${size}/"
	main='/.mounts/labs/awadallalab/private/touellette'
	modeldir="${main}/projects/TumE/analysis/models/"
	model1="evolution_11_5_Linear_17_9_7_9.027991854570908e-06_5_0.14.TASYG7N3IJR1DLN.pt"
	model2="onesubclone_15_11_Linear_15_3_7_0.0001698628401328024_6.GS3BEXB3O906DHE.pt"
	outdir="${main}/projects/TumE/analysis/G_temulator/trainsize_${size}_"

	model_dir='/.mounts/labs/awadallalab/private/touellette/projects/TumE/analysis/models/'
	model1="evolution_11_5_Linear_17_9_7_9.027991854570908e-06_5_0.14.TASYG7N3IJR1DLN.pt"
	model2="onesubclone_15_11_Linear_15_3_7_0.0001698628401328024_6.GS3BEXB3O906DHE.pt"

	echo '#!/bin/bash' > "run_temulator_search_${size}.sh"
	echo '#$ -P awadallalab' >> "run_temulator_search_${size}.sh"
	echo '#$ -cwd' >> "run_temulator_search_${size}.sh"
	echo '#$ -l h_vmem=96G' >> "run_temulator_search_${size}.sh"
	echo '#$ -l h_rt=72:00:00' >> "run_temulator_search_${size}.sh"
	echo '#$ -V' >> "run_temulator_search_${size}.sh"
	echo 'module load python/3.6' >> "run_temulator_search_${size}.sh"
	echo "source ${main}/python/bin/activate" >> "run_temulator_search_${size}.sh"
	printf "python3 ${main}/projects/TumE/bin/transfer_learning.py --path ${datadir} --model_dir ${modeldir} --model1 ${model1} --model2 ${model2} --outputdir ${outdir}" >> "run_temulator_search_${size}.sh";
done