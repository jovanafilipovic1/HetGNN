# HetGNN: Heterogeneous Graph Neural Network for Gene-Cell Dependencies

This repository contains code for training and evaluating a heterogeneous graph neural network (HetGNN) that predicts dependencies between genes and cancer cell lines.

## Repository Structure

- `Main.py`: The main entry point that loads configuration, initializes the graph, and runs the training and testing pipeline.
- `train.py`: Contains functions for model training and preparation.
- `test.py`: Contains functions for model testing, prediction, and visualization.
- `validation.py`: Contains functions for model validation during training.
- `utils.py`: Contains utility functions used by multiple modules, including data processing and matrix construction.
- `config/parameters.json`: Configuration file with all parameters for the model and training.
- `models/`: Directory containing model definitions.
- `NetworkAnalysis/`: Directory containing code for graph creation and analysis.
- `Data/`: Directory containing data files (CRISPR data, cell features, gene features, etc.).

## Key Functions

- `utils.py`: 
  - `process_data_and_create_mappings`: Processes the heterogeneous graph data and creates necessary mappings between cell lines, genes, and their indices.
  - `construct_complete_predMatrix`: Constructs a complete prediction matrix from model predictions.

- `train.py`:
  - `train_model`: Main training function for the HetGNN model.
  - `prepare_model`: Creates and initializes the model.
  - `prepare_data_for_training`: Splits the data and creates data loaders.

- `validation.py`:
  - `validate_model`: Validates the model performance on validation data.
  - `evaluate_full_predictions`: Evaluates full predictions on all genes and cell lines.

- `test.py`:
  - `test_model`: Tests the model on test data.
  - `generate_full_predictions`: Generates predictions for all gene-cell pairs.
  - `save_results`: Saves model outputs and visualizations.

## Setup and Requirements

This codebase requires:
- Python 3.8+
- PyTorch
- PyTorch Geometric
- NumPy
- pandas
- scikit-learn
- seaborn
- matplotlib

## Usage

To run the model:

```bash
python Main.py
```

This will use the configuration in `config/parameters.json` to train, validate, and test the model.

## Configuration

All parameters are defined in `config/parameters.json`. Here's what each section controls:

### Experiment Information
- `experiment_name`: Name of the experiment
- `seed`: Random seed for reproducibility
- `crp_pos`: CRISPR threshold for positive dependencies

### Data Parameters
- `cancer_type`: Cancer type to focus on (or "All" for all types)
- `base_path`: Path to the data directory
- `cell_feat_name`: Cell feature type to use
- `gene_feat_name`: Gene feature type to use
- `ppi`: Protein-protein interaction network to use
- `remove_rpl`: Whether to remove RPL genes
- `remove_commonE`: Whether to remove common essential genes
- `useSTD`: Whether to use standard deviation

### Graph Parameters
- `ppi_train_ratio`: Training ratio for PPI edges
- `metapaths`: List of metapaths to include in the graph

### Model Parameters
- `hidden_features`: Hidden layer dimensions
- `out_channels`: Output dimension
- `num_layers`: Number of GNN layers
- `dropout`: Dropout rate
- `lr`: Learning rate
- `weight_decay`: Weight decay for optimizer
- `epochs`: Number of training epochs
- `emb_dim`: Embedding dimension
- `heads`: Number of attention heads for each layer
- `lp_model`: Link prediction model type
- `gcn_model`: GNN model type
- `layer_name`: GNN layer type
- `aggregate`: Aggregation function

### Training Parameters
- `batch_size`: Training batch size
- `validation_ratio`: Ratio of data to use for validation
- `test_ratio`: Ratio of data to use for testing
- `disjoint_train_ratio`: Disjoint train ratio
- `train_neg_sampling`: Whether to sample negatives before training
- `negative_sampling_ratio`: Ratio of negative to positive samples
- `save_full_pred`: Whether to save full predictions
- `plot_cell_embeddings`: Whether to plot cell embeddings

## Output

After training, the model will save:
- Model checkpoints in `Data/NB_results/model/`
- Embeddings and predictions in `Data/NB_results/file/`
- Optional visualizations in `Data/NB_results/Figures/`

## Citation

If you use this code, please cite the corresponding paper (TODO: Add citation information).
