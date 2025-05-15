# Heterogeneous Graph Neural Network for Cancer Cell Dependency prediction

This repository contains code for training and evaluating a heterogeneous graph neural network that predicts dependencies between genes and cancer cell lines.

## Repository Structure

- `Main.py`: The main entry point that loads configuration, initializes the graph, and runs the training and testing pipeline.
- `NetworkAnalysis/Create_heterogeneous_graph.py`: Contains class for the creation of heterogeneous graph data object.
- `modules/train.py`: Contains functions for model training and preparation.
- `modules/validation.py`: Contains functions for model validation during training.
- `modules/test.py`: Contains functions for model testing, prediction, and visualization.
- `modules/utils.py`: Contains utility functions used for training and testing.
- `config/parameters.json`: Configuration file with all parameters for the model and training.
- `models/`: Directory containing models.
- `runPython.sh`: Script for running the model on HPC.
- `MOSA`: Directory for creating cell line embeddings.

## Usage

To run the model:

```bash
python Main.py --config ./config/parameters.json'
```

This will use the configuration in `config/parameters.json` to call or create the graph data object, train, validate, and test the GNN.

## Data sources

### CRISPR data and cell features: [DepMap](https://depmap.org/portal/data_page/?tab=allData)

- Common essential genes: AchillesCommonEssentialControls.csv
- Gene dependency probabilities: CRISPRGeneDependency.csv
- Cell line metadata: Model.csv
- CNV: OmicsCNGene.csv
- Expression: OmicsExpressionProteinCodingGenesTPMLogp1.csv
- Mutations: OmicsSomaticMutationsMatrixDamaging.csv

### Additional cell feature data
- MOSA cell line embeddings can be found in the directory `MOSA/reports/vae/files`. [Cai et al. (2024)](https://doi.org/10.1038/s41467-024-54771-4)
- Dependency-marker associations can be found in supplementary material of the following paper (Table S4): [Pacini et al. (2024)](https://doi.org/10.1016/j.ccell.2023.12.016)

### Gene features: [Human MsigDB Collections](https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp)
- C2: [chemical and genetic perturbations](https://www.gsea-msigdb.org/gsea/msigdb/download_file.jsp?filePath=/msigdb/release/2024.1.Hs/c2.cgp.v2024.1.Hs.symbols.gmt) (cgp)
- C4: [computational gene sets defined by mining large collections of cancer-oriented expression data](https://www.gsea-msigdb.org/gsea/msigdb/download_file.jsp?filePath=/msigdb/release/2024.1.Hs/c4.all.v2024.1.Hs.symbols.gmt)

### Protein-protein interaction network
[Reactome3.txt](https://reactome.org/download/tools/ReatomeFIs/FIsInGene_070323_with_annotations.txt.zip)

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
- `max_epochs`: Maximum number of training epochs (early stopping will occur if validation loss doesn't improve for 12 consecutive epochs)
- `emb_dim`: Embedding dimension
- `heads`: Number of attention heads for each layer as comma-separated string (e.g., "1,1")
- `lp_model`: Link prediction model type:
  - `"simple"`: Uses a dot product between node embeddings (faster, fewer parameters)
  - `"deep"`: Uses feature concatenation followed by an MLP (more expressive, more parameters)
- `gcn_model`: GNN model type (e.g., "simple")
- `aggregate`: Aggregation function (e.g., "mean")
- `cell_layer_type`: Type of layer to use for cell lines ("gnn" or "linear")

### Training Parameters
- `batch_size`: Training batch size
- `validation_ratio`: Ratio of data to use for validation
- `test_ratio`: Ratio of data to use for testing
- `disjoint_train_ratio`: Ratio of disjoint training edges

## Output

After training, the model will save:
- Model checkpoints in `Data/Results/model/`
- Embeddings and predictions in `Data/Results/file/`
- Optional visualizations in `Data/Results/Figures/`
