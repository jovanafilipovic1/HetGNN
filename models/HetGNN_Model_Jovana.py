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


class LPsimple_classif(nn.Module):
    def forward(self, x_nt1: Tensor, x_nt2: Tensor, edge_label_index: Tensor) -> Tensor:

        edge_feat_nt1 = x_nt1[edge_label_index[0]]
        edge_feat_nt2 = x_nt2[edge_label_index[1]]

        # Apply dot product for final prediction
        return (edge_feat_nt1 * edge_feat_nt2).sum(dim=-1) #this returns a vector where each element is the dot product of the corresponding edge (aka the probability of the edge)


class HeteroData_GNNmodel(nn.Module):
    def __init__(self, heterodata: HeteroData, node_types: list, node_types_to_pred: list, embedding_dim, features_dim: dict,
                 heads: list=None, dropout: float=0.2, act_fn: torch.nn.modules.activation=torch.nn.ReLU(), lp_model: str='simple',
                 aggregate: str='sum' , **kwargs):
        super().__init__()
        # We learn separate embedding matrices for each node type
        self.node_types = node_types
        self.node_types_to_pred = node_types_to_pred 

        #features.insert(0, embedding_dim)
        if isinstance(embedding_dim, int):
            self.nt1_lin = torch.nn.Linear(features_dim[node_types[0]], embedding_dim)
            self.nt2_lin = torch.nn.Linear(features_dim[node_types[1]], embedding_dim) # CHECK THIS niet nodig om zowel linear als emb te hebben ???
            # self.nt1_emb = torch.nn.Embedding(num_embeddings=heterodata[node_types[0]].num_nodes,
            #                                 embedding_dim=embedding_dim)
            # self.nt2_emb = torch.nn.Embedding(num_embeddings=heterodata[node_types[1]].num_nodes,
            #                                 embedding_dim=embedding_dim)
            
        elif isinstance(embedding_dim, dict):
            self.nt1_lin = torch.nn.Linear(features_dim[node_types[0]], embedding_dim[node_types[0]])
            self.nt2_lin = torch.nn.Linear(features_dim[node_types[1]], embedding_dim[node_types[1]])
            # self.nt1_emb = torch.nn.Embedding(num_embeddings=heterodata[node_types[0]].num_nodes,
            #                                 embedding_dim=embedding_dim[node_types[0]])
            # self.nt2_emb = torch.nn.Embedding(num_embeddings=heterodata[node_types[1]].num_nodes,
            #                                 embedding_dim=embedding_dim[node_types[1]])
            
        else:
            TypeError,"Use correct embedding dim type"

        self.conv1 = HeteroConv({('gene', 'interacts_with', 'gene'): GCNConv(-1,256),
                                       ('gene', 'rev_interacts_with', 'gene'): GCNConv(-1,256),
                                       ('cell', 'metapath_1', 'cell'): GCNConv(-1,256)
                                        }, aggr='sum')
        
        self.conv2 = HeteroConv({('gene', 'interacts_with', 'gene'): GCNConv(-1,128),
                                       ('gene', 'rev_interacts_with', 'gene'): GCNConv(-1,128),
                                       ('cell', 'metapath_1', 'cell'): GCNConv(-1,128)
                                        }, aggr='sum')
        self.act_fn = act_fn()
        self.dropout = nn.Dropout(dropout)
        
        # Apply classif of supervised edges
        if lp_model == 'simple':
            self.classifier = LPsimple_classif()
        else:
            self.classifier = LPdeep_classif(in_features=features[-1])

    def forward(self, data: HeteroData=None, edge_type_label: str=None,
                return_embeddings: bool=False, x_dict: dict=None) -> Tensor:
        etl = edge_type_label.split(',') #("gene", dependency_of", "cell")
        # x_dict holds the feature matrix of all node_types

        # x_dict = data.x_dict
        if len(self.node_types) == 2:
            x_dict = {self.node_types[0]: data[self.node_types[0]].x, 
                       self.node_types[1]: data[self.node_types[1]].x}  #You can also use x_dict = data.x_dict here!!
          
        x_dict = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {k: self.dropout(self.act_fn(v)) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, data.edge_index_dict)

        pred = self.classifier(x_dict[self.node_types_to_pred[0]],
                                x_dict[self.node_types_to_pred[1]],
                                data[etl[0], etl[1], etl[2]].edge_label_index)
        if pred.ndim == 2:
            pred = pred.ravel() #Flattening predictions
        
        if return_embeddings:
            return pred, x_dict
        else:
            return pred