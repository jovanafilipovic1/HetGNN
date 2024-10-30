#!/bin/bash
#PBS -N run_python                    ## job name
#PBS -l nodes=1:ppn=8                ## 1 node, 16 cores
#PBS -l walltime=04:00:00             ## max. 2h of wall time

module swap cluster/joltik

module load Pytorch 
moudle load numpy
module load pandas
module load wandb

echo "Module Loading Complete"

cd /data/gent/458/vsc45895/Thesis

python hetgnn_main.py --epochs 30 --lr 0.0001 --batch_size 256 --emb_dim 512 --seed 42 --exp_name example_test