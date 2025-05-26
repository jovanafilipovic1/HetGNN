import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from models.HetGNN_Model_Jovana import HeteroData_GNNmodel_Jovana
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from datetime import datetime

 # Load model and data
model_path = 'Results/Neuroblastoma_cgp_expression_marker_genes_gnn/models/Neuroblastoma_cgp_expression_marker_genes_gnn_seed37_epoch14.pt'
data_path = 'Data/multigraphs/heteroData_gene_cell_Neuroblastoma_cgp_expression_marker_genes_META2.pt'


# Configuration parameters
MODEL_CONFIG = {
    'model_type': 'gnn',  # Options: 'gnn-gnn', 'gnn-mlp', 'mlp', 'gnn'
    'embedding_dim': 512,
    'hidden_features': [-1, 512, 256],  # Adjusted to match checkpoint
    'dropout': 0.2,
    'act_fn': nn.Sigmoid,  
    'lp_model': 'deep',  # Options: 'simple', 'deep'
    'aggregate': 'mean'
}

def create_results_directory(base_path='Results/feature_analysis', model_type=MODEL_CONFIG['model_type']):
    """
    Create a directory for saving analysis results.

    Returns:
        Path to the created directory
    """
    # Create timestamp for unique directory name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(base_path, f'{model_type}_{timestamp}')
    
    # Create directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Results will be saved in: {results_dir}")
    return results_dir

def load_gene_names(base_path='Data'):
    """
    Load and filter gene names from expression data based on marker genes.
    
    Args:
        base_path: Base path for data files
        
    Returns:
        List of filtered gene names
    """
    # Load expression data
    expression_path = os.path.join(base_path, 'Depmap/OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv')
    expression = pd.read_csv(expression_path, header=0, index_col=0)
    
    # Clean gene names (remove any text after space if present)
    expression.columns = [i.split(' ')[0] for i in expression.columns]
    
    # Load marker genes from Pacini dataset
    path_mmc5 = os.path.join(base_path, 'Pacini/1-s2.0-S1535610823004440-mmc5.xls')
    df = pd.read_excel(path_mmc5, sheet_name='Extended DMAs')
    
    # Extract the desired columns
    marker_genes = df[['FEATURE', 'TARGET']]
    
    # Extract the 'FEATURES' containing '_Expr'
    expr_marker_genes = marker_genes[marker_genes['FEATURE'].str.contains('_Expr')]
    
    # Extract gene names from FEATURE column by removing "_Expr" suffix
    marker_genes = expr_marker_genes['FEATURE'].str.replace('_Expr', '')
    
    # Find the intersection of marker genes and the columns in expression data
    valid_marker_genes = marker_genes[marker_genes.isin(expression.columns)]
    unique_valid_marker_genes = valid_marker_genes.unique()
    
    print(f"Found {len(unique_valid_marker_genes)} marker genes in expression data")
    
    return list(unique_valid_marker_genes)

def load_model_and_data(model_path, data_path, device='cuda' if torch.cuda.is_available() else 'cpu', model_config=None):
    """
    Load the trained model and data.
    
    Args:
        model_path: Path to the saved model
        data_path: Path to the saved data
        device: Device to load the model on
        model_config: Optional dictionary with model configuration parameters.
                     If None, uses the global MODEL_CONFIG.
        
    Returns:
        model: Loaded model
        data: Loaded data
    """
    print(f"Loading data from: {data_path}")
    # Load data
    data = torch.load(data_path)
    print("Data loaded successfully")
    print(f"Data structure: {data}")
    print(f"Available edge types: {data.edge_types}")
    
    # Define node types and feature dimensions
    node_types = ['gene', 'cell']
    features_dim = {
        'gene': data['gene'].x.shape[1],
        'cell': data['cell'].x.shape[1]
    }
    print(f"Feature dimensions: {features_dim}")
    
    # Use provided config or fall back to global config
    config = model_config if model_config is not None else MODEL_CONFIG
    
    # Create model with same architecture as saved model
    print(f"Loading model from: {model_path}")
    model = HeteroData_GNNmodel_Jovana(
        heterodata=data,
        node_types=node_types,
        node_types_to_pred=node_types,
        embedding_dim=config['embedding_dim'],
        features_dim=features_dim,
        hidden_features=config['hidden_features'],
        dropout=config['dropout'],
        act_fn=config['act_fn'],
        lp_model=config['lp_model'],
        aggregate=config['aggregate'],
        model_type=config['model_type']
    )
    
    # Load trained weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model weights loaded successfully")
    except Exception as e:
        print(f"Error loading model weights: {str(e)}")
        raise
    
    model.to(device)
    model.eval()
    
    return model, data

def compute_feature_importance(model, data, device, num_samples=1000):
    """
    Compute feature importance using gradients for any model architecture.
    
    Args:
        model: Trained model (any architecture from HetGNN_Model_Jovana)
        data: HeteroData object
        device: Device to run computation on
        num_samples: Number of samples to use for averaging
        
    Returns:
        Dictionary with feature importance scores for cell features
    """
    print("Computing feature importance...")
    
    # Initialize importance scores
    cell_importance = torch.zeros(data['cell'].x.shape[1], device=device)
    
    # Get edge indices from the data
    try:
        edge_index = data['gene', 'dependency_of', 'cell'].edge_index
        print(f"Found {edge_index.shape[1]} edges")
    except Exception as e:
        print(f"Error accessing edge indices: {str(e)}")
        print("Available edge types:", data.edge_types)
        raise
    
    num_edges = edge_index.shape[1]
    indices = torch.randperm(num_edges)[:num_samples]
    print(f"Using {num_samples} random samples for feature importance computation")
    
    for i, idx in enumerate(indices):
        if i % 100 == 0:
            print(f"Processing sample {i}/{num_samples}")
            
        # Get the specific gene-cell pair
        gene_idx = edge_index[0, idx]
        cell_idx = edge_index[1, idx]
        
        # Create a copy of the data
        data_copy = data.clone()
        
        # Enable gradient computation for input features
        data_copy['cell'].x.requires_grad_(True)
        
        # Create edge_label_index for this specific edge
        edge_label_index = torch.stack([gene_idx.unsqueeze(0), cell_idx.unsqueeze(0)], dim=0)
        data_copy['gene', 'dependency_of', 'cell'].edge_label_index = edge_label_index
        
        # Forward pass - this works for all model architectures
        out = model(data_copy, edge_type_label="gene,dependency_of,cell")
        
        # Get prediction for this specific edge
        pred = out[0]  # Since we only have one edge in edge_label_index
        
        # Compute gradients
        pred.backward()
        
        # Accumulate absolute gradients
        cell_importance += data_copy['cell'].x.grad[cell_idx].abs()
        
        # Reset gradients
        data_copy['cell'].x.grad = None
    
    # Average the importance scores
    cell_importance /= num_samples
    
    print("Feature importance computation completed")
    return {
        'cell_importance': cell_importance.cpu().numpy()
    }

def analyze_feature_importance(importance_scores, gene_names, top_k=20):
    """
    Analyze and visualize feature importance scores.
    
    Args:
        importance_scores: Dictionary with importance scores
        gene_names: List of gene names
        top_k: Number of top features to display
    """
    # For cell features
    cell_importance = importance_scores['cell_importance']
    cell_indices = np.argsort(cell_importance)[-top_k:]
    
    plt.figure(figsize=(12, 6))
    plt.barh(range(top_k), cell_importance[cell_indices])
    plt.yticks(range(top_k), [gene_names[i] for i in cell_indices])
    plt.title('Top {} Most Important Cell Features'.format(top_k))
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show()
    
    # Print top features
    print("\nTop {} Most Important Cell Features:".format(top_k))
    for idx in cell_indices:
        print(f"{gene_names[idx]}: {cell_importance[idx]:.4f}")

def save_importance_scores(importance_scores, gene_names, output_dir):
    """
    Save feature importance scores to CSV files and create summary plots.
    
    Args:
        importance_scores: Dictionary with importance scores
        gene_names: List of gene names
        output_dir: Directory to save the results
    """
    # Create DataFrame for cell features
    cell_df = pd.DataFrame({
        'gene_name': gene_names,
        'importance_score': importance_scores['cell_importance']
    })
    cell_df = cell_df.sort_values('importance_score', ascending=False)
    
    # Save to CSV
    cell_df.to_csv(os.path.join(output_dir, 'cell_features_importance.csv'), index=False)
    
    # Create and save plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=cell_df.head(20), x='importance_score', y='gene_name')
    plt.title('Top 20 Most Important Cell Features')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_cell_features.png'))
    plt.close()
    
    # Save summary statistics
    with open(os.path.join(output_dir, 'analysis_summary.txt'), 'w') as f:
        f.write("Feature Importance Analysis Summary\n")
        f.write("=================================\n\n")
        
        f.write("Cell Features Summary:\n")
        f.write(f"Total number of features: {len(cell_df)}\n")
        f.write(f"Mean importance score: {cell_df['importance_score'].mean():.4f}\n")
        f.write(f"Max importance score: {cell_df['importance_score'].max():.4f}\n")
        f.write(f"Min importance score: {cell_df['importance_score'].min():.4f}\n\n")
        
        f.write("Top 10 Cell Features:\n")
        for _, row in cell_df.head(10).iterrows():
            f.write(f"{row['gene_name']}: {row['importance_score']:.4f}\n")

if __name__ == "__main__":
    # Create results directory
    results_dir = create_results_directory(model_type=MODEL_CONFIG['model_type'])
    
   
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, data = load_model_and_data(model_path, data_path, device)
    
    # Load gene names
    gene_names = load_gene_names()
    
    # Compute feature importance
    importance_scores = compute_feature_importance(model, data, device)
    
    # Analyze and visualize results
    analyze_feature_importance(importance_scores, gene_names)
    
    # Save results
    save_importance_scores(importance_scores, gene_names, results_dir)
    
    print(f"\nAnalysis complete! Results have been saved in: {results_dir}")
    print("The following files have been created:")
    print("1. cell_features_importance.csv - Complete cell feature importance scores")
    print("2. top_cell_features.png - Visualization of top 20 cell features")
    print("3. analysis_summary.txt - Summary statistics and top features")