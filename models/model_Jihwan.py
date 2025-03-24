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

gnn_factory = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GATv2": geom_nn.GATv2Conv,
    "GraphConv": geom_nn.GraphConv,
    "sageconv": geom_nn.SAGEConv,
}


class GCNsimple(nn.Module):
    def __init__(self, hidden_channels: list, layer_name:str, aggregate:str, **kwargs) -> None:
        super().__init__()
        gnnlayer = gnn_factory[layer_name]
        
        # Add add_self_loops=False only for layers that support it (GCN, GAT)
        layer_kwargs = {'aggr': aggregate}
        if layer_name in ['GCN', 'GAT', 'GATv2']:
            layer_kwargs['add_self_loops'] = False
            
        self.gnn1 = gnnlayer(in_channels=hidden_channels[0], out_channels=hidden_channels[1], **layer_kwargs)
        self.gnn2 = gnnlayer(in_channels=hidden_channels[1], out_channels=hidden_channels[2], **layer_kwargs)
        # self.gnn3 = gnnlayer(in_channels=hidden_channels[2], out_channels=hidden_channels[3], aggr = aggregate, **kwargs)


    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:

        x = self.gnn1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.6)

        # x = F.relu(self.gnn2(x, edge_index))
        # x = F.dropout(x, training=self.training, p=0.2)

        # x = F.relu(self.gnn3(x, edge_index))
        # x = F.dropout(x, training=self.training, p=0.2)

        x = self.gnn2(x, edge_index)
        return x

class GATsimple(nn.Module):
    def __init__(self, hidden_channels: list, layer_name:str, **kwargs) -> None:
        super().__init__()

        gnnlayer = gnn_factory[layer_name]
        # Add add_self_loops=False only for layers that support it
        layer_kwargs = {}
        if layer_name in ['GCN', 'GAT', 'GATv2']:
            layer_kwargs['add_self_loops'] = False
            
        self.gnn1 = gnnlayer(in_channels=hidden_channels[0], out_channels=hidden_channels[1], **layer_kwargs)
        self.gnn2 = gnnlayer(in_channels=hidden_channels[1], out_channels=hidden_channels[2], **layer_kwargs)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:

        # x, attw1 = self.gnn1(x=x, edge_index=edge_index, return_attention_weights=True)
        x, (indices, att_w) = self.gnn1(x=x, edge_index=edge_index, return_attention_weights=True)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.2)

        x, (indices, att_w) = self.gnn2(x=x, edge_index=edge_index, return_attention_weights=True)
        return x

class GCNcustom(nn.Module):
    def __init__(self, features: list, layer_name: str='sageconv', heads: list=None,
                 dropout: float=0.2, act_fn: torch.nn.modules.activation=torch.nn.ReLU,
                   **kwargs) -> None:
        super().__init__(**kwargs)

        self.dropout = dropout
        self.num_layers = len(features)-1 # embedding dimension
        if layer_name == 'GAT':
            assert self.num_layers == len(features)-1 == len(heads),"Wrong paramter sizes"
            self.return_attention_weights = kwargs.get("return_attention_weights", True)
            self.heads = [1] + heads
        else:
            self.return_attention_weights = False

        # Define layers

        if layer_name == 'GAT':
            layers = [gat_layer(in_features=features[i]*self.heads[i], hidden_features=features[i+1],
                            act_fn=act_fn, dropout=self.dropout, heads=self.heads[i+1], ix=i,
                            layer_name=layer_name, layer=gnn_factory[layer_name], add_self_loops=False)
                  for i in range(self.num_layers)]

            layers = OrderedDict(chain(*[i.items() for i in layers]))
            # self.gnn_layers = geom_nn.Sequential('x, edge_index, return_attention_weights', layers)
            self.gnn_layers = geom_nn.Sequential('x, edge_index', layers)
        else:
            layers = [gat_layer(in_features=features[i], hidden_features=features[i+1],
                            act_fn=act_fn, dropout=self.dropout, heads=None, ix=i,
                            layer_name=layer_name, layer=gnn_factory[layer_name])
                  for i in range(self.num_layers)]

            layers = OrderedDict(chain(*[i.items() for i in layers]))
            self.gnn_layers = geom_nn.Sequential('x, edge_index', layers)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        embeddings = self.gnn_layers(x, edge_index)

        return embeddings

class LPsimple_classif(nn.Module):

    # for now only do dot product so only forward pass
    def forward(self, x_nt1: Tensor, x_nt2: Tensor, edge_label_index: Tensor) -> Tensor:

        edge_feat_nt1 = x_nt1[edge_label_index[0]]
        edge_feat_nt2 = x_nt2[edge_label_index[1]]

        # Apply dot product for final prediction
        return (edge_feat_nt1 * edge_feat_nt2).sum(dim=-1) #this returns a vector where each element is the dot product of the corresponding edge (aka the probability of the edge)

class LPdeep_classif(nn.Module):
    def __init__(self, in_features) -> None:
        super().__init__()
        self.lin_layers = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(in_features=64, out_features=1)) #reduces each edge representation to a single scalar output (score) per edge.
    def forward(self, x_nt1: Tensor, x_nt2: Tensor, edge_label_index: Tensor) -> Tensor:

        edge_feat_nt1 = x_nt1[edge_label_index[0]]
        edge_feat_nt2 = x_nt2[edge_label_index[1]]

        edge_ = (edge_feat_nt1 * edge_feat_nt2)

        return self.lin_layers(edge_)  #returns vector with the score/probability of each edge

class HeteroData_GNNmodel_Jihwan(nn.Module):
    def __init__(self, heterodata: HeteroData, node_types: list, node_types_to_pred: list, embedding_dim, features_dim: dict,
                 gcn_model: str, features: list, layer_name: str='sageconv', heads: list=None,
                 dropout: float=0.2, act_fn: torch.nn.modules.activation=torch.nn.ReLU, lp_model: str='simple',
                 aggregate: str='mean' , **kwargs):
        super().__init__()
        # We learn separate embedding matrices for each node type
        self.node_types = node_types
        self.node_types_to_pred = node_types_to_pred 
        self.gcn_model = gcn_model

        #features.insert(0, embedding_dim)
        if isinstance(embedding_dim, int):
            self.nt1_lin = torch.nn.Linear(features_dim[node_types[0]], embedding_dim)
            self.nt2_lin = torch.nn.Linear(features_dim[node_types[1]], embedding_dim) # CHECK THIS niet nodig om zowel linear als emb te hebben ???
            self.nt1_emb = torch.nn.Embedding(num_embeddings=heterodata[node_types[0]].num_nodes,
                                            embedding_dim=embedding_dim)
            self.nt2_emb = torch.nn.Embedding(num_embeddings=heterodata[node_types[1]].num_nodes,
                                            embedding_dim=embedding_dim)
            if len(node_types) == 3: 
                self.nt3_lin = torch.nn.Linear(features_dim[node_types[2]], embedding_dim)
                self.nt3_emb = torch.nn.Embedding(num_embeddings=heterodata[node_types[2]].num_nodes,
                                                  embedding_dim=embedding_dim)
        elif isinstance(embedding_dim, dict):
            self.nt1_lin = torch.nn.Linear(features_dim[node_types[0]], embedding_dim[node_types[0]])
            self.nt2_lin = torch.nn.Linear(features_dim[node_types[1]], embedding_dim[node_types[1]])
            self.nt1_emb = torch.nn.Embedding(num_embeddings=heterodata[node_types[0]].num_nodes,
                                            embedding_dim=embedding_dim[node_types[0]])
            self.nt2_emb = torch.nn.Embedding(num_embeddings=heterodata[node_types[1]].num_nodes,
                                            embedding_dim=embedding_dim[node_types[1]])
            if len(node_types) == 3:
                self.nt3_lin = torch.nn.Linear(features_dim[node_types[2]], embedding_dim[node_types[2]])
                self.nt3_emb = torch.nn.Embedding(num_embeddings=heterodata[node_types[2]].num_nodes,
                                                  embedding_dim=embedding_dim[node_types[2]])
        else:
            TypeError,"Use correct embedding dim type"

        # Instantiate homoGNN
        if self.gcn_model == 'simple':
            self.gnn = GCNsimple(hidden_channels=features, layer_name=layer_name, aggregate=aggregate, **kwargs)
            self.gnn = geom_nn.to_hetero(self.gnn, metadata=heterodata.metadata())

        elif self.gcn_model == 'gat':
            self.gnn = GATsimple(hidden_channels=features, layer_name=layer_name, **kwargs) ##waarom niet omgezet naar heteroGNN ????
        else:
            self.gnn = GCNcustom(features=features,
                                 layer_name=layer_name, heads=heads,
                                 dropout=dropout, act_fn=act_fn) # CHECK REST OF THIS FUNCTION
            self.gnn = geom_nn.to_hetero(self.gnn.gnn_layers, metadata=heterodata.metadata())

        # Convert to heteroGNN

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
            # x_dict = {self.node_types[0]: self.nt1_emb(data[self.node_types[0]].node_id),
            #           self.node_types[1]: self.nt2_emb(data[self.node_types[1]].node_id)}
            # x_dict = {self.node_types[0]: self.nt1_lin(data[self.node_types[0]].x),
            #           self.node_types[1]: self.nt2_lin(data[self.node_types[1]].x)}
            # x_dict = {self.node_types[0]: self.nt1_emb(data[self.node_types[0]].node_id) + self.nt1_lin(data[self.node_types[0]].x),
            #           self.node_types[1]: self.nt2_emb(data[self.node_types[1]].node_id) + self.nt2_lin(data[self.node_types[1]].x)}
            x_dict = {self.node_types[0]: data[self.node_types[0]].x, 
                       self.node_types[1]: data[self.node_types[1]].x}  #You can also use x_dict = data.x_dict here!!
            # x_dict = {self.node_types[0]: data[self.node_types[0]].x,
            #           self.node_types[1]: self.nt2_emb(data[self.node_types[1]].node_id)}
            # x_dict = {self.node_types[0]: data[self.node_types[0]].x,
            #           self.node_types[1]: self.nt2_emb(data[self.node_types[1]].node_id) + self.nt2_lin(data[self.node_types[1]].x)}
            # x_dict = {self.node_types[0]:  self.nt1_lin(data[self.node_types[0]].x) + self.nt1_emb(data[self.node_types[0]].node_id),
            #            self.node_types[1]: self.nt2_lin(data[self.node_types[1]].x) + self.nt2_emb(data[self.node_types[1]].node_id)}
        # else:
            # x_dict = {self.node_types[0]: self.nt1_emb(data[self.node_types[0]].node_id),
            #           self.node_types[1]: self.nt2_emb(data[self.node_types[1]].node_id),
            #           self.node_types[2]: self.nt3_emb(data[self.node_types[2]].node_id)}
            # x_dict = {self.node_types[0]: self.nt1_lin(data[self.node_types[0]].x),
            #           self.node_types[1]: self.nt2_lin(data[self.node_types[1]].x),
            #           self.node_types[2]: self.nt3_lin(data[self.node_types[2]].x)}
            # x_dict = {self.node_types[0]: self.nt1_emb(data[self.node_types[0]].node_id) + self.nt1_lin(data[self.node_types[0]].x),
            #           self.node_types[1]: self.nt2_emb(data[self.node_types[1]].node_id) + self.nt2_lin(data[self.node_types[1]].x),
            #           self.node_types[2]: self.nt3_emb(data[self.node_types[2]].node_id) + self.nt3_lin(data[self.node_types[2]].x)}
            # x_dict = {self.node_types[0]: data[self.node_types[0]].x,
            #           self.node_types[1]: data[self.node_types[1]].x,
            #           self.node_types[2]: data[self.node_types[2]].x}

        if self.gcn_model == 'simple':
            x_dict = self.gnn(x_dict, data.edge_index_dict)
        else:
            x_dict = self.gnn(x_dict, data.edge_index_dict)

        pred = self.classifier(x_dict[self.node_types_to_pred[0]],
                                x_dict[self.node_types_to_pred[1]],
                                data[etl[0], etl[1], etl[2]].edge_label_index)
        if pred.ndim == 2:
            pred = pred.ravel() #Why?

        if self.gcn_model == 'gat:': #redundant ?!
            if return_embeddings:
                return pred, x_dict 
            else:
                return pred
        if return_embeddings:
            return pred, x_dict
        else:
            return pred
        
