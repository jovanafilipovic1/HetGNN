import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.data as geom_data
import torch_geometric.nn as geom_nn
from typing import Optional, Dict, List, Union, Tuple, Any
from torch import Tensor
from itertools import chain
from collections import OrderedDict
from torch_geometric.data import HeteroData, Data
from torch_geometric.nn import GCNConv, GATv2Conv, HeteroConv, SAGEConv
import numpy as np


class LPsimple_classif(nn.Module):
    """Simple dot product classifier for link prediction."""
    def forward(self, x_nt1: Tensor, x_nt2: Tensor, edge_label_index: Tensor) -> Tensor:
        edge_feat_nt1 = x_nt1[edge_label_index[0]] 
        edge_feat_nt2 = x_nt2[edge_label_index[1]]

        # Apply dot product for final prediction
        return (edge_feat_nt1 * edge_feat_nt2).sum(dim=-1)


class LPdeep_classif(nn.Module):
    """Deep MLP classifier for link prediction."""
    def __init__(self, in_features, hidden_dim=64):
        """
        Initialize a deep classifier that concatenates node features and passes them through an MLP.
        
        Args:
            in_features: The feature dimension of each node embedding
            hidden_dim: Hidden dimension of the MLP
        """
        super().__init__()
        # MLP layers after concatenation
        self.mlp = nn.Sequential(
            nn.Linear(in_features * 2, hidden_dim),  # Multiply by 2 because we concatenate two node embeddings
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x_nt1: Tensor, x_nt2: Tensor, edge_label_index: Tensor) -> Tensor:
        # Get node features for the edges
        edge_feat_nt1 = x_nt1[edge_label_index[0]]
        edge_feat_nt2 = x_nt2[edge_label_index[1]]
        
        # Concatenate the features
        concat_features = torch.cat([edge_feat_nt1, edge_feat_nt2], dim=-1)
        
        # Apply MLP and return predictions
        pred = self.mlp(concat_features)
        return pred.squeeze(-1)  # Remove the last dimension to match the same shape as LPsimple_classif


class BaseHetGNNModel(nn.Module):
    """Base class for all HetGNN models."""
    def __init__(
        self, 
        heterodata: HeteroData, 
        node_types: List[str], 
        node_types_to_pred: List[str],
        embedding_dim: Union[int, Dict[str, int]], 
        features_dim: Dict[str, int],
        hidden_features: List[int] = [-1, 256, 128],
        dropout: float = 0.2, 
        act_fn: torch.nn.Module = torch.nn.ReLU(), 
        lp_model: str = 'simple',
        **kwargs
    ):
        super().__init__()
        self.node_types = node_types
        self.node_types_to_pred = node_types_to_pred
        self.act_fn = act_fn()
        self.dropout = nn.Dropout(dropout)
        
        # Parse hidden features
        self.hidden_features = hidden_features
        self.hidden_dim1 = self.hidden_features[1] if len(self.hidden_features) > 1 else 256
        self.hidden_dim2 = self.hidden_features[2] if len(self.hidden_features) > 2 else 128
        
        # Initialize the embedding layers
        if isinstance(embedding_dim, int):
            self.nt1_lin = torch.nn.Linear(features_dim[node_types[0]], embedding_dim)
            self.nt2_lin = torch.nn.Linear(features_dim[node_types[1]], embedding_dim)
        elif isinstance(embedding_dim, dict):
            self.nt1_lin = torch.nn.Linear(features_dim[node_types[0]], embedding_dim[node_types[0]])
            self.nt2_lin = torch.nn.Linear(features_dim[node_types[1]], embedding_dim[node_types[1]])
        else:
            raise TypeError("Use correct embedding dim type: int or dict")
        
        # Store lp_model type for later initialization in derived classes
        self.lp_model_type = lp_model
    
    def forward(self, data: HeteroData = None, edge_type_label: str = None,
                return_embeddings: bool = False, x_dict: dict = None) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        """Implement in derived classes."""
        raise NotImplementedError("Base class does not implement forward")


class GNN_GNN_Model(BaseHetGNNModel):
    """Model where both gene and cell nodes use GNN message passing."""
    def __init__(
        self, 
        heterodata: HeteroData, 
        node_types: List[str], 
        node_types_to_pred: List[str],
        embedding_dim: Union[int, Dict[str, int]], 
        features_dim: Dict[str, int],
        hidden_features: List[int] = [-1, 256, 128],
        dropout: float = 0.2, 
        act_fn: torch.nn.Module = torch.nn.ReLU(), 
        lp_model: str = 'simple',
        aggregate: str = 'sum',
        **kwargs
    ):
        super().__init__(
            heterodata=heterodata,
            node_types=node_types,
            node_types_to_pred=node_types_to_pred,
            embedding_dim=embedding_dim,
            features_dim=features_dim,
            hidden_features=hidden_features,
            dropout=dropout,
            act_fn=act_fn,
            lp_model=lp_model
        )
        
        # Add projection layers for gene skip connections
        if isinstance(embedding_dim, int):
            self.gene_skip_proj1 = torch.nn.Linear(embedding_dim, self.hidden_dim1)
            self.gene_skip_proj2 = torch.nn.Linear(self.hidden_dim1, self.hidden_dim2)
            self.cell_skip_proj1 = torch.nn.Linear(embedding_dim, self.hidden_dim1)
            self.cell_skip_proj2 = torch.nn.Linear(self.hidden_dim1, self.hidden_dim2)
        else:
            self.gene_skip_proj1 = torch.nn.Linear(embedding_dim[node_types[0]], self.hidden_dim1)
            self.gene_skip_proj2 = torch.nn.Linear(self.hidden_dim1, self.hidden_dim2)
            self.cell_skip_proj1 = torch.nn.Linear(embedding_dim[node_types[1]], self.hidden_dim1)
            self.cell_skip_proj2 = torch.nn.Linear(self.hidden_dim1, self.hidden_dim2)
        
        # Define GNN layers
        # First convolutional layer for gene-gene and cell-cell interactions
        self.conv1 = HeteroConv({
            ('gene', 'interacts_with', 'gene'): GATv2Conv(-1, self.hidden_dim1),
            ('gene', 'rev_interacts_with', 'gene'): GATv2Conv(-1, self.hidden_dim1),
            ('cell', 'metapath_0', 'cell'): GCNConv(-1, self.hidden_dim1)
        }, aggr=aggregate)
        
        # Second convolutional layer
        self.conv2 = HeteroConv({
            ('gene', 'interacts_with', 'gene'): GATv2Conv(-1, self.hidden_dim2),
            ('gene', 'rev_interacts_with', 'gene'): GATv2Conv(-1, self.hidden_dim2),
            ('cell', 'metapath_0', 'cell'): GCNConv(-1, self.hidden_dim2)
        }, aggr=aggregate)
        
        # Initialize classifier based on lp_model type
        if self.lp_model_type == 'simple':
            self.classifier = LPsimple_classif()
            print(f"Using simple dot product classifier for link prediction")
        elif self.lp_model_type == 'deep':
            self.classifier = LPdeep_classif(in_features=self.hidden_dim2)
            print(f"Using deep MLP classifier for link prediction")
        else:
            raise ValueError(f"Unknown lp_model: {self.lp_model_type}. Must be 'simple' or 'deep'")
    
    def forward(self, data: HeteroData = None, edge_type_label: str = None,
                return_embeddings: bool = False, x_dict: dict = None) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        etl = edge_type_label.split(',')
        
        # Apply initial linear transformations
        x_dict = {
            self.node_types[0]: self.nt1_lin(data[self.node_types[0]].x),  # gene embedding
            self.node_types[1]: self.nt2_lin(data[self.node_types[1]].x)   # cell embedding
        }
        
        # Store initial embeddings for skip connections
        initial_gene_embeddings = x_dict[self.node_types[0]].clone()
        initial_cell_embeddings = x_dict[self.node_types[1]].clone()
        
        # First layer processing with skip connections
        layer1_output = self.conv1(x_dict, data.edge_index_dict, edge_attr_dict=data.edge_attr_dict)
        
        # Apply skip connections
        projected_cell_skip1 = self.cell_skip_proj1(initial_cell_embeddings)
        layer1_output['cell'] = layer1_output['cell'] + projected_cell_skip1
        
        projected_gene_skip1 = self.gene_skip_proj1(initial_gene_embeddings)
        layer1_output['gene'] = layer1_output['gene'] + projected_gene_skip1
        
        x_dict = layer1_output
        
        # Store embeddings after the first layer for second layer skip connection
        cell_layer1_output = x_dict['cell'].clone()
        gene_layer1_output = x_dict['gene'].clone()
        
        # Apply activation and dropout to all node types
        x_dict = {k: self.dropout(self.act_fn(v)) for k, v in x_dict.items()}
        
        # Second layer processing with skip connections
        layer2_output = self.conv2(x_dict, data.edge_index_dict, edge_attr_dict=data.edge_attr_dict)
        
        # Apply skip connections
        projected_cell_skip2 = self.cell_skip_proj2(cell_layer1_output)
        layer2_output['cell'] = layer2_output['cell'] + projected_cell_skip2
        
        projected_gene_skip2 = self.gene_skip_proj2(gene_layer1_output)
        layer2_output['gene'] = layer2_output['gene'] + projected_gene_skip2
        
        x_dict = layer2_output
        
        # Make prediction
        pred = self.classifier(x_dict[self.node_types_to_pred[0]],
                              x_dict[self.node_types_to_pred[1]],
                              data[etl[0], etl[1], etl[2]].edge_label_index)
        
        if pred.ndim == 2:
            pred = pred.ravel()  # Flattening predictions
        
        if return_embeddings:
            return pred, x_dict
        else:
            return pred


class GNN_MLP_Model(BaseHetGNNModel):
    """Model where gene nodes use GNN message passing and cell nodes use MLP."""
    def __init__(
        self, 
        heterodata: HeteroData, 
        node_types: List[str], 
        node_types_to_pred: List[str],
        embedding_dim: Union[int, Dict[str, int]], 
        features_dim: Dict[str, int],
        hidden_features: List[int] = [-1, 256, 128],
        dropout: float = 0.2, 
        act_fn: torch.nn.Module = torch.nn.ReLU(), 
        lp_model: str = 'simple',
        aggregate: str = 'sum',
        **kwargs
    ):
        super().__init__(
            heterodata=heterodata,
            node_types=node_types,
            node_types_to_pred=node_types_to_pred,
            embedding_dim=embedding_dim,
            features_dim=features_dim,
            hidden_features=hidden_features,
            dropout=dropout,
            act_fn=act_fn,
            lp_model=lp_model
        )
        
        # Add projection layers for gene skip connections
        if isinstance(embedding_dim, int):
            self.gene_skip_proj1 = torch.nn.Linear(embedding_dim, self.hidden_dim1)
            self.gene_skip_proj2 = torch.nn.Linear(self.hidden_dim1, self.hidden_dim2)
            
            # Linear layers for cell lines
            self.cell_lin1 = torch.nn.Linear(embedding_dim, self.hidden_dim1)
            self.cell_lin2 = torch.nn.Linear(self.hidden_dim1, self.hidden_dim2)
            # Skip connection projections for cell linear layers
            self.cell_skip_proj1_linear = torch.nn.Linear(embedding_dim, self.hidden_dim1)
            self.cell_skip_proj2_linear = torch.nn.Linear(self.hidden_dim1, self.hidden_dim2)
        else:
            self.gene_skip_proj1 = torch.nn.Linear(embedding_dim[node_types[0]], self.hidden_dim1)
            self.gene_skip_proj2 = torch.nn.Linear(self.hidden_dim1, self.hidden_dim2)
            
            # Linear layers for cell lines
            self.cell_lin1 = torch.nn.Linear(embedding_dim[node_types[1]], self.hidden_dim1)
            self.cell_lin2 = torch.nn.Linear(self.hidden_dim1, self.hidden_dim2)
            # Skip connection projections for cell linear layers
            self.cell_skip_proj1_linear = torch.nn.Linear(embedding_dim[node_types[1]], self.hidden_dim1)
            self.cell_skip_proj2_linear = torch.nn.Linear(self.hidden_dim1, self.hidden_dim2)
        
        # Define GNN layers for genes only
        # First convolutional layer for gene-gene interactions
        self.conv1 = HeteroConv({
            ('gene', 'interacts_with', 'gene'): GATv2Conv(-1, self.hidden_dim1),
            ('gene', 'rev_interacts_with', 'gene'): GATv2Conv(-1, self.hidden_dim1)
        }, aggr=aggregate)
        
        # Second convolutional layer for genes
        self.conv2 = HeteroConv({
            ('gene', 'interacts_with', 'gene'): GATv2Conv(-1, self.hidden_dim2),
            ('gene', 'rev_interacts_with', 'gene'): GATv2Conv(-1, self.hidden_dim2)
        }, aggr=aggregate)
        
        # Initialize classifier based on lp_model type
        if self.lp_model_type == 'simple':
            self.classifier = LPsimple_classif()
            print(f"Using simple dot product classifier for link prediction")
        elif self.lp_model_type == 'deep':
            self.classifier = LPdeep_classif(in_features=self.hidden_dim2)
            print(f"Using deep MLP classifier for link prediction")
        else:
            raise ValueError(f"Unknown lp_model: {self.lp_model_type}. Must be 'simple' or 'deep'")
    
    def forward(self, data: HeteroData = None, edge_type_label: str = None,
                return_embeddings: bool = False, x_dict: dict = None) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        etl = edge_type_label.split(',')
        
        # Apply initial linear transformations
        x_dict = {
            self.node_types[0]: self.nt1_lin(data[self.node_types[0]].x),  # gene embedding
            self.node_types[1]: self.nt2_lin(data[self.node_types[1]].x)   # cell embedding
        }
        
        # Store initial embeddings for skip connections
        initial_gene_embeddings = x_dict[self.node_types[0]].clone()
        initial_cell_embeddings = x_dict[self.node_types[1]].clone()
        
        # First layer processing - GNN for genes, MLP for cells
        # Process genes with GNN
        gene_x_dict = {'gene': x_dict['gene']}
        gene_x_dict = self.conv1(gene_x_dict, data.edge_index_dict, edge_attr_dict=data.edge_attr_dict)
        
        # Apply skip connection for genes
        projected_gene_skip1 = self.gene_skip_proj1(initial_gene_embeddings)
        x_dict['gene'] = gene_x_dict['gene'] + projected_gene_skip1
        
        # Process cells with linear layer
        linear_output = self.cell_lin1(x_dict['cell'])
        # Apply skip connection for cells
        projected_initial_embeddings = self.cell_skip_proj1_linear(initial_cell_embeddings)
        x_dict['cell'] = linear_output + projected_initial_embeddings
        
        # Store embeddings after the first layer for second layer skip connection
        cell_layer1_output = x_dict['cell'].clone()
        gene_layer1_output = x_dict['gene'].clone()
        
        # Apply activation and dropout to all node types
        x_dict = {k: self.dropout(self.act_fn(v)) for k, v in x_dict.items()}
        
        # Second layer processing - GNN for genes, MLP for cells
        # Process genes with GNN
        gene_x_dict = {'gene': x_dict['gene']}
        gene_x_dict = self.conv2(gene_x_dict, data.edge_index_dict, edge_attr_dict=data.edge_attr_dict)
        
        # Apply skip connection for genes
        projected_gene_skip2 = self.gene_skip_proj2(gene_layer1_output)
        x_dict['gene'] = gene_x_dict['gene'] + projected_gene_skip2
        
        # Process cells with linear layer
        linear_output = self.cell_lin2(x_dict['cell'])
        # Apply skip connection for cells
        projected_layer1_output = self.cell_skip_proj2_linear(cell_layer1_output)
        x_dict['cell'] = linear_output + projected_layer1_output
        
        # Make prediction
        pred = self.classifier(x_dict[self.node_types_to_pred[0]],
                              x_dict[self.node_types_to_pred[1]],
                              data[etl[0], etl[1], etl[2]].edge_label_index)
        
        if pred.ndim == 2:
            pred = pred.ravel()  # Flattening predictions
        
        if return_embeddings:
            return pred, x_dict
        else:
            return pred


class MLP_Model(BaseHetGNNModel):
    """
    Model where node embeddings are processed only by a simple initial transformation
    and then a deep classifier is used on concatenated embeddings.
    """
    def __init__(
        self, 
        heterodata: HeteroData, 
        node_types: List[str], 
        node_types_to_pred: List[str],
        embedding_dim: Union[int, Dict[str, int]], 
        features_dim: Dict[str, int],
        dropout: float = 0.2, 
        act_fn: torch.nn.Module = torch.nn.ReLU(), 
        lp_model: str = 'deep',  # MLP model always uses deep classifier
        **kwargs
    ):
        super().__init__(
            heterodata=heterodata,
            node_types=node_types,
            node_types_to_pred=node_types_to_pred,
            embedding_dim=embedding_dim,
            features_dim=features_dim,
            dropout=dropout,
            act_fn=act_fn,
            lp_model=lp_model
        )
        
        # For MLP model, determine embedding dimension
        if isinstance(embedding_dim, int):
            self.emb_dim = embedding_dim
        else:
            # For MLP model, we need the same dimension for both node types
            if embedding_dim[node_types[0]] != embedding_dim[node_types[1]]:
                raise ValueError("For MLP model, gene and cell embedding dimensions must be the same")
            self.emb_dim = embedding_dim[node_types[0]]
        
        # MLP model always uses a deep classifier
        # If lp_model was set to 'simple', override it with a warning
        if self.lp_model_type == 'simple':
            print("Warning: MLP model requires a deep classifier. Overriding lp_model='simple' with 'deep'.")
        
        # Initialize deep classifier directly
        self.classifier = LPdeep_classif(in_features=self.emb_dim)
        print(f"Using deep MLP classifier for link prediction in MLP model")
    
    def forward(self, data: HeteroData = None, edge_type_label: str = None,
                return_embeddings: bool = False, x_dict: dict = None) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        etl = edge_type_label.split(',')
        
        # Apply initial linear transformations
        x_dict = {
            self.node_types[0]: self.nt1_lin(data[self.node_types[0]].x),  # gene embedding
            self.node_types[1]: self.nt2_lin(data[self.node_types[1]].x)   # cell embedding
        }
        
        # Apply dropout to embeddings
        x_dict = {k: self.dropout(v) for k, v in x_dict.items()}
        
        # Make prediction using deep classifier
        pred = self.classifier(x_dict[self.node_types_to_pred[0]],
                              x_dict[self.node_types_to_pred[1]],
                              data[etl[0], etl[1], etl[2]].edge_label_index)
        
        if pred.ndim == 2:
            pred = pred.ravel()  # Flattening predictions
        
        if return_embeddings:
            return pred, x_dict
        else:
            return pred


class GNN_Model(BaseHetGNNModel):
    """
    Model of Jihwan, where gene and cell communicate directly with each other via "dependency_of" and "rev_dependency_of" edges.
    Genes communicate with each other via "interacts_with" and "rev_interacts_with" edges.
    """
    def __init__(
        self, 
        heterodata: HeteroData, 
        node_types: List[str], 
        node_types_to_pred: List[str],
        embedding_dim: Union[int, Dict[str, int]], 
        features_dim: Dict[str, int],
        hidden_features: List[int] = [-1, 256, 128],
        dropout: float = 0.2, 
        act_fn: torch.nn.Module = torch.nn.ReLU(), 
        lp_model: str = 'simple',
        aggregate: str = 'sum',
        **kwargs
    ):
        super().__init__(
            heterodata=heterodata,
            node_types=node_types,
            node_types_to_pred=node_types_to_pred,
            embedding_dim=embedding_dim,
            features_dim=features_dim,
            hidden_features=hidden_features,
            dropout=dropout,
            act_fn=act_fn,
            lp_model=lp_model
        )
        
        # Add projection layers for skip connections
        if isinstance(embedding_dim, int):
            self.gene_skip_proj1 = torch.nn.Linear(embedding_dim, self.hidden_dim1)
            self.gene_skip_proj2 = torch.nn.Linear(self.hidden_dim1, self.hidden_dim2)
            self.cell_skip_proj1 = torch.nn.Linear(embedding_dim, self.hidden_dim1)
            self.cell_skip_proj2 = torch.nn.Linear(self.hidden_dim1, self.hidden_dim2)
        else:
            self.gene_skip_proj1 = torch.nn.Linear(embedding_dim[node_types[0]], self.hidden_dim1)
            self.gene_skip_proj2 = torch.nn.Linear(self.hidden_dim1, self.hidden_dim2)
            self.cell_skip_proj1 = torch.nn.Linear(embedding_dim[node_types[1]], self.hidden_dim1)
            self.cell_skip_proj2 = torch.nn.Linear(self.hidden_dim1, self.hidden_dim2)
        
        # Define GNN layers
        # First convolutional layer for different edge types
        self.conv1 = HeteroConv({
            # Gene-gene interaction with GAT
            ('gene', 'interacts_with', 'gene'): GATv2Conv(-1, self.hidden_dim1),
            ('gene', 'rev_interacts_with', 'gene'): GATv2Conv(-1, self.hidden_dim1),
            
            # Gene-cell and cell-gene interaction with GCN
            ('gene', 'dependency_of', 'cell'): SAGEConv(-1, self.hidden_dim1),
            ('cell', 'rev_dependency_of', 'gene'): SAGEConv(-1, self.hidden_dim1)
        }, aggr=aggregate)
        
        # Second convolutional layer
        self.conv2 = HeteroConv({
            # Gene-gene interaction with GAT
            ('gene', 'interacts_with', 'gene'): GATv2Conv(-1, self.hidden_dim2),
            ('gene', 'rev_interacts_with', 'gene'): GATv2Conv(-1, self.hidden_dim2),
            
            # Gene-cell and cell-gene interaction with GCN
            ('gene', 'dependency_of', 'cell'): SAGEConv(-1, self.hidden_dim2),
            ('cell', 'rev_dependency_of', 'gene'): SAGEConv(-1, self.hidden_dim2)
        }, aggr=aggregate)
        
        # Initialize classifier based on lp_model type
        if self.lp_model_type == 'simple':
            self.classifier = LPsimple_classif()
            print(f"Using simple dot product classifier for link prediction")
        elif self.lp_model_type == 'deep':
            self.classifier = LPdeep_classif(in_features=self.hidden_dim2)
            print(f"Using deep MLP classifier for link prediction")
        else:
            raise ValueError(f"Unknown lp_model: {self.lp_model_type}. Must be 'simple' or 'deep'")
    
    def forward(self, data: HeteroData = None, edge_type_label: str = None,
                return_embeddings: bool = False, x_dict: dict = None) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        etl = edge_type_label.split(',')
        
        # Apply initial linear transformations
        x_dict = {
            self.node_types[0]: self.nt1_lin(data[self.node_types[0]].x),  # gene embedding
            self.node_types[1]: self.nt2_lin(data[self.node_types[1]].x)   # cell embedding
        }
        
        # Store initial embeddings for skip connections
        initial_gene_embeddings = x_dict[self.node_types[0]].clone()
        initial_cell_embeddings = x_dict[self.node_types[1]].clone()
        
        # First layer processing with skip connections
        layer1_output = self.conv1(x_dict, data.edge_index_dict, edge_attr_dict=data.edge_attr_dict)
        
        # Apply skip connections
        projected_cell_skip1 = self.cell_skip_proj1(initial_cell_embeddings)
        layer1_output['cell'] = layer1_output['cell'] + projected_cell_skip1
        
        projected_gene_skip1 = self.gene_skip_proj1(initial_gene_embeddings)
        layer1_output['gene'] = layer1_output['gene'] + projected_gene_skip1
        
        x_dict = layer1_output
        
        # Store embeddings after the first layer for second layer skip connection
        cell_layer1_output = x_dict['cell'].clone()
        gene_layer1_output = x_dict['gene'].clone()
        
        # Apply activation and dropout to all node types
        x_dict = {k: self.dropout(self.act_fn(v)) for k, v in x_dict.items()}
        
        # Second layer processing with skip connections
        layer2_output = self.conv2(x_dict, data.edge_index_dict, edge_attr_dict=data.edge_attr_dict)
        
        # Apply skip connections
        projected_cell_skip2 = self.cell_skip_proj2(cell_layer1_output)
        layer2_output['cell'] = layer2_output['cell'] + projected_cell_skip2
        
        projected_gene_skip2 = self.gene_skip_proj2(gene_layer1_output)
        layer2_output['gene'] = layer2_output['gene'] + projected_gene_skip2
        
        x_dict = layer2_output
        
        # Make prediction
        pred = self.classifier(x_dict[self.node_types_to_pred[0]],
                              x_dict[self.node_types_to_pred[1]],
                              data[etl[0], etl[1], etl[2]].edge_label_index)
        
        if pred.ndim == 2:
            pred = pred.ravel()  # Flattening predictions
        
        if return_embeddings:
            return pred, x_dict
        else:
            return pred


def HeteroData_GNNmodel_Jovana(
    heterodata: HeteroData, 
    node_types: List[str], 
    node_types_to_pred: List[str], 
    embedding_dim: Union[int, Dict[str, int]], 
    features_dim: Dict[str, int],
    hidden_features: List[int] = [-1, 256, 128],
    dropout: float = 0.2, 
    act_fn: torch.nn.Module = torch.nn.ReLU(), 
    lp_model: str = 'simple',
    aggregate: str = 'sum',
    model_type: str = 'gnn-gnn',
    **kwargs
) -> nn.Module:
    """
    Factory function to create the appropriate model based on the model type parameter.
    
    Args:
        heterodata: The heterogeneous graph data
        node_types: List of node types in the graph
        node_types_to_pred: List of node types to predict links between
        embedding_dim: Dimension for initial node embeddings (int or dict mapping node types to dimensions)
        features_dim: Dictionary mapping node types to their feature dimensions
        hidden_features: List of hidden feature dimensions, should be in format [input_dim, hidden_dim1, hidden_dim2]
        dropout: Dropout rate
        act_fn: Activation function
        lp_model: Link prediction model ('simple' for dot product, 'deep' for MLP)
        aggregate: Aggregation method for message passing ('sum', 'mean', etc.)
        model_type: Type of model architecture to use ('gnn-gnn', 'gnn-mlp', 'mlp', or 'gnn')
        
    Returns:
        Initialized model of the specified type
    """
    print(f"Creating model of type: {model_type}")
    
    model_type = model_type.lower()
    
    if model_type == 'gnn-gnn':
        return GNN_GNN_Model(
            heterodata=heterodata,
            node_types=node_types,
            node_types_to_pred=node_types_to_pred,
            embedding_dim=embedding_dim,
            features_dim=features_dim,
            hidden_features=hidden_features,
            dropout=dropout,
            act_fn=act_fn,
            lp_model=lp_model,
            aggregate=aggregate,
            **kwargs
        )
    elif model_type == 'gnn-mlp':
        return GNN_MLP_Model(
            heterodata=heterodata,
            node_types=node_types,
            node_types_to_pred=node_types_to_pred,
            embedding_dim=embedding_dim,
            features_dim=features_dim,
            hidden_features=hidden_features,
            dropout=dropout,
            act_fn=act_fn,
            lp_model=lp_model,
            aggregate=aggregate,
            **kwargs
        )
    elif model_type == 'mlp':
        return MLP_Model(
            heterodata=heterodata,
            node_types=node_types,
            node_types_to_pred=node_types_to_pred,
            embedding_dim=embedding_dim,
            features_dim=features_dim,
            dropout=dropout,
            act_fn=act_fn,
            lp_model=lp_model,  # This will be overridden to 'deep' inside the MLP_Model
            **kwargs
        )
    elif model_type == 'gnn':
        return GNN_Model(
            heterodata=heterodata,
            node_types=node_types,
            node_types_to_pred=node_types_to_pred,
            embedding_dim=embedding_dim,
            features_dim=features_dim,
            hidden_features=hidden_features,
            dropout=dropout,
            act_fn=act_fn,
            lp_model=lp_model,
            aggregate=aggregate,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Must be one of: 'gnn-gnn', 'gnn-mlp', 'mlp', 'gnn'")
    
