#!/usr/bin/env bash
#SBATCH --mincpus 8
#SBATCH --mem 20000
#SBATCH --exclude rockford,steele,hammer,conan,blomquist,wolfe,knatterton,holmes,lenssen,scuderi,matula,marlowe,poirot,monk

##SBATCH -LXserver
##SBATCH --exclude turbine,vane
##SBATCH --gres gpu:1


srun -u xvfb-run -a python remote_experiment.py "$@"
#srun -u python remote_experiment.py "$@"
