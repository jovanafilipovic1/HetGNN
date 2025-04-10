#!/bin/bash
#PBS -N run_python                   ## job name
#PBS -l nodes=1:ppn=3:gpus=1         ## 1 node, 16 cores
#PBS -l walltime=48:00:00            ## max. hours of wall time (increased for grid search)
#PBS -l mem=16gb                     ## Increased memory for grid search
#PBS -m abe                          ## e-mail notifications (abe=aborted, begin and end)

export PATH=$PATH:/user/gent/458/vsc45895/.local/bin

# load CUDA and Python module first
module load CUDA-Python/12.1.0-gfbf-2023a-CUDA-12.1.1

# check GCCcore module version
#module list GCCcore

# we require CMake for installing pyg from source; afterwards we will not need it
module load CMake/3.26.3-GCCcore-12.3.0

# check cuda version
#nvcc --version

# install pytorch for the desired cuda version, generate URL at https://pytorch.org/get-started/locally/
#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#python -c "import torch; print(torch.version.cuda); print(torch.__version__)"

# install torch geometric
#pip3 install torch_geometric
#python -c "import torch_geometric; print(torch_geometric.__version__)"

# install torch_sparse and torch_scatter for the right torch and cuda version
#pip3 install torch_sparse torch_scatter -f https://pytorch-geometric.com/whl/torch-2.5.1+cu121.html
#python -c "import torch_sparse; print(torch_sparse.__version__)"
#python -c "import torch_scatter; print(torch_scatter.__version__)"

# install pyg_lib from master
#pip install ninja wheel
#pip install git+https://github.com/pyg-team/pyg-lib.git
#python -c "import pyg_lib; print(pyg_lib.__version__)"


module load scikit-learn/1.3.1-gfbf-2023a
module load Seaborn/0.13.2-gfbf-2023a
module load matplotlib/3.7.2-gfbf-2023a
#module load wandb/0.16.1-GCC-12.3.0

echo "Module Loading Complete"

cd /data/gent/458/vsc45895/Thesis

python3 Main.py --config  config/parametersgnn.json --grid_search    