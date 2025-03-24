import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.data as geom_data
import torch_geometric.nn as geom_nn
from typing import Optional
from torch import Tensor
from itertools import chain
from collections import OrderedDict
from torch_geometric.data import HeteroData, Data
from torch_geometric.nn import GCNConv, GATv2Conv, HeteroConv
import numpy as np

# Define the utility function for embedding analysis
def print_cell_embedding_stats(x_dict, node_type='cell', tag='', sample_indices=None):
    """
    Print statistics about cell embeddings and sample values for specific indices.
    
    Args:
        x_dict: Dictionary of node embeddings
        node_type: Node type to analyze (default: 'cell')
        tag: Identifier string for the print statements
        sample_indices: List of specific indices to print (default: None)
    """
    if node_type not in x_dict:
        print(f"{tag} - {node_type} embeddings not found in dictionary")
        return
    
    embeddings = x_dict[node_type]
    
    # Calculate statistics
    emb_mean = embeddings.mean(dim=0).mean().item()
    emb_std = embeddings.std(dim=0).mean().item()
    emb_min = embeddings.min().item()
    emb_max = embeddings.max().item()
    
    print(f"\n{tag} - {node_type} embedding stats:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Mean: {emb_mean:.6f}, Std: {emb_std:.6f}, Min: {emb_min:.6f}, Max: {emb_max:.6f}")
    
    # Print sample embeddings for specific indices
    if sample_indices is not None and len(sample_indices) > 0:
        print(f"  Sample {node_type} embeddings for indices {sample_indices}:")
        for idx in sample_indices:
            if idx < embeddings.shape[0]:
                # Print first 5 values and last 5 values
                values = embeddings[idx].detach().cpu().numpy()
                if len(values) > 10:
                    sample_str = np.array2string(values[:5], precision=4, suppress_small=True) + " ... " + \
                                np.array2string(values[-5:], precision=4, suppress_small=True)
                else:
                    sample_str = np.array2string(values, precision=4, suppress_small=True)
                print(f"    Index {idx}: {sample_str}")
                
                # Calculate variance for this specific embedding
                emb_var = embeddings[idx].var().item()
                print(f"    Variance: {emb_var:.6f}")
                
                # Check for NaN or Inf values
                if torch.isnan(embeddings[idx]).any() or torch.isinf(embeddings[idx]).any():
                    print(f"    WARNING: NaN or Inf values detected in embedding for index {idx}")
            else:
                print(f"    Index {idx} out of bounds")

    # Calculate pairwise cosine similarity between the sample embeddings
    if sample_indices is not None and len(sample_indices) > 1:
        from torch.nn.functional import cosine_similarity
        print("  Pairwise cosine similarities between sample embeddings:")
        for i, idx1 in enumerate(sample_indices):
            if idx1 >= embeddings.shape[0]:
                continue
            for j, idx2 in enumerate(sample_indices[i+1:], i+1):
                if idx2 >= embeddings.shape[0]:
                    continue
                sim = cosine_similarity(
                    embeddings[idx1].unsqueeze(0),
                    embeddings[idx2].unsqueeze(0)
                ).item()
                print(f"    Similarity between {idx1} and {idx2}: {sim:.6f}")


class LPsimple_classif(nn.Module):
    def forward(self, x_nt1: Tensor, x_nt2: Tensor, edge_label_index: Tensor) -> Tensor:

        edge_feat_nt1 = x_nt1[edge_label_index[0]]
        edge_feat_nt2 = x_nt2[edge_label_index[1]]

        # Apply dot product for final prediction
        return (edge_feat_nt1 * edge_feat_nt2).sum(dim=-1) #this returns a vector where each element is the dot product of the corresponding edge (aka the probability of the edge)


class HeteroData_GNNmodel_Jovana(nn.Module):
    def __init__(self, heterodata: HeteroData, node_types: list, node_types_to_pred: list, embedding_dim, features_dim: dict,
                 heads: list=None, dropout: float=0.2, act_fn: torch.nn.modules.activation=torch.nn.ReLU(), lp_model: str='simple',
                 aggregate: str='sum', cell_layer_type: str='GNN', **kwargs):
        super().__init__()
        # We learn separate embedding matrices for each node type
        self.node_types = node_types
        self.node_types_to_pred = node_types_to_pred 
        
        # Sample indices to track for debugging (first 4 cells and genes)
        self.debug_cell_indices = [0, 1, 2, 3]
        self.debug_gene_indices = [0, 1, 2, 3] 
        self.debug_epoch = 0
        
        # Store the cell layer type
        self.cell_layer_type = cell_layer_type.lower()
        print(f"Using {self.cell_layer_type.upper()} layers for cell lines")
        print(f"Debug mode enabled - will print detailed cell embedding stats during epoch 1")
        print(f"Tracking cell indices: {self.debug_cell_indices}")
        print(f"ADDED SKIP CONNECTIONS to prevent homogeneity in cell embeddings")
        print(f"ADDED SKIP CONNECTIONS to prevent homogeneity in gene embeddings")
        
        # Define dimensions for the hidden layers
        self.hidden_dim1 = 256
        self.hidden_dim2 = 128
        
        # Initialize linear layers for each node type
        if isinstance(embedding_dim, int):
            self.nt1_lin = torch.nn.Linear(features_dim[node_types[0]], embedding_dim)
            self.nt2_lin = torch.nn.Linear(features_dim[node_types[1]], embedding_dim)
            
            # Additional linear layers for cell lines if using linear mode
            if self.cell_layer_type == 'linear':
                self.cell_lin1 = torch.nn.Linear(embedding_dim, self.hidden_dim1)
                self.cell_lin2 = torch.nn.Linear(self.hidden_dim1, self.hidden_dim2)
                # Skip connection projections for linear mode
                self.cell_skip_proj1_linear = torch.nn.Linear(embedding_dim, self.hidden_dim1)
                self.cell_skip_proj2_linear = torch.nn.Linear(self.hidden_dim1, self.hidden_dim2)
                
            # Add projection layers for gene skip connections
            self.gene_skip_proj1 = torch.nn.Linear(embedding_dim, self.hidden_dim1)
            self.gene_skip_proj2 = torch.nn.Linear(self.hidden_dim1, self.hidden_dim2)
                
        elif isinstance(embedding_dim, dict):
            self.nt1_lin = torch.nn.Linear(features_dim[node_types[0]], embedding_dim[node_types[0]])
            self.nt2_lin = torch.nn.Linear(features_dim[node_types[1]], embedding_dim[node_types[1]])
            
            # Additional linear layers for cell lines if using linear mode
            if self.cell_layer_type == 'linear':
                self.cell_lin1 = torch.nn.Linear(embedding_dim[node_types[1]], self.hidden_dim1)
                self.cell_lin2 = torch.nn.Linear(self.hidden_dim1, self.hidden_dim2)
                # Skip connection projections for linear mode
                self.cell_skip_proj1_linear = torch.nn.Linear(embedding_dim[node_types[1]], self.hidden_dim1)
                self.cell_skip_proj2_linear = torch.nn.Linear(self.hidden_dim1, self.hidden_dim2)
                
            # Add projection layers for gene skip connections
            self.gene_skip_proj1 = torch.nn.Linear(embedding_dim[node_types[0]], self.hidden_dim1)
            self.gene_skip_proj2 = torch.nn.Linear(self.hidden_dim1, self.hidden_dim2)
        else:
            TypeError,"Use correct embedding dim type"

        # Define GNN layers
        # First convolutional layer - gene-gene interactions are always message passing
        conv1_dict = {
            ('gene', 'interacts_with', 'gene'): GATv2Conv(-1, self.hidden_dim1),
            ('gene', 'rev_interacts_with', 'gene'): GATv2Conv(-1, self.hidden_dim1)
        }
        
        # Add cell-cell metapath only if using GNN for cells
        if self.cell_layer_type == 'gnn':
            conv1_dict[('cell', 'metapath_0', 'cell')] = GCNConv(-1, self.hidden_dim1)
            
        self.conv1 = HeteroConv(conv1_dict, aggr=aggregate)
        
        # Second convolutional layer
        conv2_dict = {
            ('gene', 'interacts_with', 'gene'): GATv2Conv(-1, self.hidden_dim2),
            ('gene', 'rev_interacts_with', 'gene'): GATv2Conv(-1, self.hidden_dim2)
        }
        
        # Add cell-cell metapath only if using GNN for cells
        if self.cell_layer_type == 'gnn':
            conv2_dict[('cell', 'metapath_0', 'cell')] = GCNConv(-1, self.hidden_dim2)
            
        self.conv2 = HeteroConv(conv2_dict, aggr=aggregate)
        
        # Add projection layers for skip connections in GNN mode
        if self.cell_layer_type == 'gnn':
            # For first skip connection (input embeddings → hidden_dim1)
            if isinstance(embedding_dim, int):
                self.cell_skip_proj1 = torch.nn.Linear(embedding_dim, self.hidden_dim1)
            else:
                self.cell_skip_proj1 = torch.nn.Linear(embedding_dim[node_types[1]], self.hidden_dim1)
            
            # For second skip connection (hidden_dim1 → hidden_dim2)
            self.cell_skip_proj2 = torch.nn.Linear(self.hidden_dim1, self.hidden_dim2)
        
        self.act_fn = act_fn()
        self.dropout = nn.Dropout(dropout)
        
        # Apply classif of supervised edges
        if lp_model == 'simple':
            self.classifier = LPsimple_classif()
        else:
            self.classifier = LPdeep_classif(in_features=features[-1])

    def set_debug_epoch(self, epoch):
        """Set the current epoch for debugging purposes"""
        self.debug_epoch = epoch
        if epoch == 1:
            print(f"\n==== Entering epoch {epoch} - DEBUG OUTPUT ENABLED ====")
        elif epoch == 0:
            print(f"\n==== Starting training - Epoch {epoch} ====")
      
    def forward(self, data: HeteroData=None, edge_type_label: str=None,
                return_embeddings: bool=False, x_dict: dict=None) -> Tensor:
        etl = edge_type_label.split(',')
        
        if len(self.node_types) == 2:
            # Apply initial linear transformations
            x_dict = {
                self.node_types[0]: self.nt1_lin(data[self.node_types[0]].x),  # gene
                self.node_types[1]: self.nt2_lin(data[self.node_types[1]].x)   # cell
            }
            
            # Store initial embeddings for skip connections
            initial_gene_embeddings = x_dict[self.node_types[0]].clone()
            initial_cell_embeddings = x_dict[self.node_types[1]].clone()
            
            # Debug: Print initial embeddings stats - only in epoch 1
            if self.training and self.debug_epoch == 1:
                print(f"\n--- Epoch {self.debug_epoch} - Initial Embeddings after linear projection ---")
                print_cell_embedding_stats(x_dict, node_type='cell', 
                                          tag='Initial', 
                                          sample_indices=self.debug_cell_indices)
                print_cell_embedding_stats(x_dict, node_type='gene', 
                                          tag='Initial', 
                                          sample_indices=self.debug_gene_indices)
        
        # First layer processing
        if self.cell_layer_type == 'gnn':
            # Use the HeteroConv with the data's edge attributes for both genes and cells
            layer1_output = self.conv1(x_dict, data.edge_index_dict, edge_attr_dict=data.edge_attr_dict)
            
            # Apply skip connection for cells - project initial embeddings and add
            projected_cell_skip1 = self.cell_skip_proj1(initial_cell_embeddings)
            layer1_output['cell'] = layer1_output['cell'] + projected_cell_skip1
            
            # Apply skip connection for genes - project initial embeddings and add
            projected_gene_skip1 = self.gene_skip_proj1(initial_gene_embeddings)
            layer1_output['gene'] = layer1_output['gene'] + projected_gene_skip1
            
            x_dict = layer1_output
        else:
            # Use HeteroConv for genes but linear layer for cells
            gene_x_dict = {'gene': x_dict['gene']}
            gene_x_dict = self.conv1(gene_x_dict, data.edge_index_dict, edge_attr_dict=data.edge_attr_dict)
            
            # Apply skip connection for genes
            projected_gene_skip1 = self.gene_skip_proj1(initial_gene_embeddings)
            x_dict['gene'] = gene_x_dict['gene'] + projected_gene_skip1
            
            # Apply linear layer for cells + skip connection
            linear_output = self.cell_lin1(x_dict['cell'])
            # Project initial cell embeddings to match hidden_dim1 before adding
            projected_initial_embeddings = self.cell_skip_proj1_linear(initial_cell_embeddings)
            x_dict['cell'] = linear_output + projected_initial_embeddings  # Skip connection
        
        # Store embeddings after the first layer for second layer skip connection
        cell_layer1_output = x_dict['cell'].clone()
        gene_layer1_output = x_dict['gene'].clone()
        
        # Debug: Print embeddings after first layer - only in epoch 1
        if self.training and self.debug_epoch == 1:
            print(f"\n--- Epoch {self.debug_epoch} - After First Layer (before activation) ---")
            print_cell_embedding_stats(x_dict, node_type='cell', 
                                      tag='After First Layer', 
                                      sample_indices=self.debug_cell_indices)
            print_cell_embedding_stats(x_dict, node_type='gene', 
                                      tag='After First Layer', 
                                      sample_indices=self.debug_gene_indices)
        
        # Apply activation and dropout to all node types
        x_dict = {k: self.dropout(self.act_fn(v)) for k, v in x_dict.items()}
        
        # Debug: Print embeddings after activation and dropout - only in epoch 1
        if self.training and self.debug_epoch == 1:
            print(f"\n--- Epoch {self.debug_epoch} - After First Layer Activation & Dropout ---")
            print_cell_embedding_stats(x_dict, node_type='cell', 
                                      tag='After First Layer + Act + Drop', 
                                      sample_indices=self.debug_cell_indices)
            print_cell_embedding_stats(x_dict, node_type='gene', 
                                      tag='After First Layer + Act + Drop', 
                                      sample_indices=self.debug_gene_indices)
        
        # Second layer processing
        if self.cell_layer_type == 'gnn':
            # Use the HeteroConv with the data's edge attributes for both genes and cells
            layer2_output = self.conv2(x_dict, data.edge_index_dict, edge_attr_dict=data.edge_attr_dict)
            
            # Apply skip connection for cells - project layer1 output and add
            projected_cell_skip2 = self.cell_skip_proj2(cell_layer1_output)
            layer2_output['cell'] = layer2_output['cell'] + projected_cell_skip2
            
            # Apply skip connection for genes - project layer1 output and add
            projected_gene_skip2 = self.gene_skip_proj2(gene_layer1_output)
            layer2_output['gene'] = layer2_output['gene'] + projected_gene_skip2
            
            x_dict = layer2_output
        else:
            # Use HeteroConv for genes but linear layer for cells
            gene_x_dict = {'gene': x_dict['gene']}
            gene_x_dict = self.conv2(gene_x_dict, data.edge_index_dict, edge_attr_dict=data.edge_attr_dict)
            
            # Apply skip connection for genes
            projected_gene_skip2 = self.gene_skip_proj2(gene_layer1_output)
            x_dict['gene'] = gene_x_dict['gene'] + projected_gene_skip2
            
            # Apply linear layer for cells + skip connection
            linear_output = self.cell_lin2(x_dict['cell'])
            # Project layer1 output to match hidden_dim2 before adding
            projected_layer1_output = self.cell_skip_proj2_linear(cell_layer1_output)
            x_dict['cell'] = linear_output + projected_layer1_output  # Skip connection from layer 1
        
        # Debug: Print embeddings after second layer - only in epoch 1
        if self.training and self.debug_epoch == 1:
            print(f"\n--- Epoch {self.debug_epoch} - Final Embeddings ---")
            print_cell_embedding_stats(x_dict, node_type='cell', 
                                      tag='Final', 
                                      sample_indices=self.debug_cell_indices)
            print_cell_embedding_stats(x_dict, node_type='gene', 
                                      tag='Final', 
                                      sample_indices=self.debug_gene_indices)
        
        pred = self.classifier(x_dict[self.node_types_to_pred[0]],
                                x_dict[self.node_types_to_pred[1]],
                                data[etl[0], etl[1], etl[2]].edge_label_index)
        if pred.ndim == 2:
            pred = pred.ravel() #Flattening predictions
        
        if return_embeddings:
            return pred, x_dict
        else:
            return pred