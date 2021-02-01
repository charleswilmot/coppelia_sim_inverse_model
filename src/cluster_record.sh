#!/usr/bin/env bash
#SBATCH --mincpus 10
#SBATCH --mem 50000
#SBATCH --exclude rockford,steele,hammer,conan,blomquist,wolfe,knatterton,holmes,lenssen,scuderi,matula,marlowe,poirot,monk
#SBATCH -LXserver
#SBATCH --gres gpu:1


srun -u python remote_record.py "$@"
