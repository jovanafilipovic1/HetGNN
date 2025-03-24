#!/bin/bash
#PBS -N run_python                   ## job name
#PBS -l nodes=1:ppn=2:gpus=1         ## 1 node, 16 cores
#PBS -l walltime=00:30:00            ## max. hours of wall time
#PBS -l mem=8gb                    ## If not used, memory will be available proportional to the max amount
#PBS -m abe                          ## e-mail notifications (abe=aborted, begin and end)

export PATH=$PATH:/user/gent/458/vsc45895/.local/bin

# load CUDA and Python module first
module load CUDA-Python/12.1.0-gfbf-2023a-CUDA-12.1.1
module load scikit-learn/1.3.1-gfbf-2023a
module load Seaborn/0.13.2-gfbf-2023a
module load matplotlib/3.7.2-gfbf-2023a
module load scanpy/1.9.8-foss-2023a #natsort
#module load Python-bundle-PyPI/2023.10-GCCcore-13.2.0
module load adjustText/1.1.1-foss-2023a
module load umap-learn/0.5.5-foss-2023a
module load CMake/3.26.3-GCCcore-12.3.0
#pip install git+https://github.com/ZhaoxiangSimonCai/shap.git
#pip install git+https://github.com/TylerYep/torchinfo
#pip install git+https://github.com/KevinMusgrave/pytorch-metric-learning
#module load PyYAML/6.0.1-GCCcore-13.2.0
#module load R/4.3.2-gfbf-2023a
pip install git+https://github.com/yaml/pyyaml


echo "Module Loading Complete"

cd /data/gent/458/vsc45895/Thesis/PhenPred

python3 PhenPred/vae/Main.py