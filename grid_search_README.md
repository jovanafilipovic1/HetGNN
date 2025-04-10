# Grid Search for HetGNN

This document explains how to use the grid search functionality for hyperparameter tuning of the heterogeneous graph neural network (HetGNN) model.

## Overview

Grid search is implemented to find the best hyperparameters for the HetGNN model. The implementation:

1. Loads a specified experiment configuration file
2. Generates all combinations of hyperparameters from the grid
3. Trains models with each combination 
4. Tracks the best model based on validation AP score
5. Saves results, including embeddings and predictions from the best model

## Configuration Files

Configuration files are stored in the `config/` directory. Each file contains:

- Basic experiment settings
- Graph parameters
- Model parameters
- Training parameters
- A hyperparameter grid for grid search

Example configuration files:
- `config/neuroblastoma_experiment.json`: Settings for Neuroblastoma-specific model
- `config/all_cancers_experiment.json`: Settings for model using all cancer types

## Hyperparameter Grid

The grid search supports the following hyperparameters:
- `learning_rate`: Model learning rate
- `batch_size`: Training batch size
- `hidden_features`: Hidden layer dimensions (as a string, e.g., "-1,256,128")
- `max_grad_norm`: Maximum gradient norm for gradient clipping

Example grid:
```json
"hyperparam_grid": {
  "learning_rate": [0.0001, 0.0005, 0.001],
  "batch_size": [64, 128, 256],
  "hidden_features": ["-1,128,64", "-1,256,128", "-1,512,256"],
  "max_grad_norm": [0.5, 1.0, 1.5]
}
```

## Running Grid Search

### Locally

```bash
python Main.py --grid_search --experiment <experiment_name>
```

Example:
```bash
python Main.py --grid_search --experiment neuroblastoma_experiment
```

### On HPC Cluster

1. Edit the `grid_search_job.pbs` file to set the experiment name:
   ```bash
   EXPERIMENT="neuroblastoma_experiment"
   ```

2. Submit the job:
   ```bash
   qsub grid_search_job.pbs
   ```

Alternatively, use the `runPython.sh` script directly:
```bash
./runPython.sh --grid_search --experiment <experiment_name>
```

## Output

Grid search results are saved in:
```
Data/NB_results/experiment_<experiment_name>/
```

Within this directory:
- `grid_search/`: Contains results for all hyperparameter combinations
  - `grid_results.csv`: CSV file with metrics for all combinations
  - `best_hyperparams.json`: Best hyperparameters found
  - `errors.log`: Any errors encountered during grid search
- `best_model/`: Contains only the results of the best model
  - `model/`: The best model checkpoint
  - `file/`: Embeddings and predictions CSV files
  - `Figures/`: Visualizations (if enabled)

## Creating New Experiments

To create a new experiment:

1. Create a new JSON file in the `config/` directory:
   ```bash
   cp config/neuroblastoma_experiment.json config/my_experiment.json
   ```

2. Edit the parameters and hyperparameter grid as needed

3. Run the grid search with your new configuration:
   ```bash
   python Main.py --grid_search --experiment my_experiment
   ```

## Best Practices

- Start with smaller grids to test the pipeline
- For large grid searches, increase the PBS job walltime and memory limits
- If your dataset is very large, consider reducing the number of combinations
- The grid search shuffles combinations to distribute the computation more evenly 