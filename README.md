# HetGNN: Heterogeneous Graph Neural Network Cancer Cell Depndency prediction

This repository contains code for training and evaluating a heterogeneous graph neural network (HetGNN) that predicts dependencies between genes and cancer cell lines.

## Repository Structure

- `Main.py`: The main entry point that loads configuration, initializes the graph, and runs the training and testing pipeline.
- `modules/train.py`: Contains functions for model training and preparation.
- `modules/validation.py`: Contains functions for model validation during training.
- `modules/test.py`: Contains functions for model testing, prediction, and visualization.
- `modules/utils.py`: Contains utility functions used by multiple modules, including data processing and matrix construction.
- `config/parameters.json`: Configuration file with all parameters for the model and training.
- `models/`: Directory containing models
- `NetworkAnalysis/Create_heterogeneous_graph.py`: Contains class for the creation of heterogeneous graph data object.
- `Data/`: Directory containing data files (CRISPR data, cell features, gene features, etc.).
- `runPython.sh`: script for running the model on HPC

## Usage

To run the model:

```bash
python Main.py
```

This will use the configuration in `config/parameters.json` to train, validate, and test the model.

## Configuration

All parameters are defined in `config/parameters.json`. Here's what each section controls:

### Settings
- `experiment_name`: Name of the experiment
- `seed`: Random seed for reproducibility
- `save_full_predictions`: Whether to save full predictions
- `plot_cell_embeddings`: Whether to plot cell embeddings using t-SNE

### Graph Parameters
- `cancer_type`: Cancer type to focus on (e.g., "Neuroblastoma" or "All" for all types)
- `base_path`: Path to the data directory
- `cell_feat_name`: Cell feature type to use (e.g., "cnv")
- `gene_feat_name`: Gene feature type to use (e.g., "cgp")
- `ppi`: Protein-protein interaction network to use
- `crp_pos`: CRISPR threshold for positive dependencies (default: -1.5)
- `metapaths`: List of metapaths to include in the graph (e.g., ["gene_cell_gene", "cell_gene_cell"])

### Model Parameters
- `hidden_features`: Hidden layer dimensions as comma-separated string (e.g., "-1,256,128")
- `dropout`: Dropout rate
- `lr`: Learning rate
- `epochs`: Number of training epochs
- `emb_dim`: Embedding dimension
- `heads`: Number of attention heads for each layer as comma-separated string (e.g., "1,1")
- `lp_model`: Link prediction model type (e.g., "simple")
- `gcn_model`: GNN model type (e.g., "simple")
- `aggregate`: Aggregation function (e.g., "mean")

### Training Parameters
- `batch_size`: Training batch size
- `validation_ratio`: Ratio of data to use for validation
- `test_ratio`: Ratio of data to use for testing
- `disjoint_train_ratio`: Ratio of disjoint training edges

## Output

After training, the model will save:
- Model checkpoints in `Data/NB_results/model/`
- Embeddings and predictions in `Data/NB_results/file/`
- Optional visualizations in `Data/NB_results/Figures/`
