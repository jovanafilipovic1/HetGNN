#from gat_dependency.utils import read_h5py
import seaborn as sns
import matplotlib.pyplot as plt
from torch_geometric.loader.link_neighbor_loader import LinkNeighborLoader
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import torch.functional as F
import torch
import pickle 
#from models.model_Jihwan import HeteroData_GNNmodel
from models.HetGNN_Model_Jovana import HeteroData_GNNmodel
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import os
import argparse
#import wandb
from datetime import datetime
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
from torch_geometric import seed_everything
from copy import deepcopy
from NetworkAnalysis import Create_heterogeneous_graph 
import json


print("Imported all libraries")


if __name__ == "__main__":

    seed_everything(37)

    gpu_available = torch.cuda.is_available()
    print(f"GPU Available: {gpu_available}")
    if gpu_available:
        device = 'cuda:1'
    else:
        device = 'cpu'

    # Load configuration from the JSON file
    with open('config/parameters.json', 'r') as config_file:
        config = json.load(config_file)

    #Extract parameters 
    print(config["experiment_name"])
    data_params = config['data']
    graph_params = config['graph']
    model_params = config['model']
    training_params = config['training']

    #read in (or make) heterodata_object
    try:
        filepath = os.path.join(
            data_params(BASE_PATH),
            'multigraphs',
            f"heteroData_gene_cell_{data_params(cancer_type).replace(' ', '_') if data_params(cancer_type) else 'All'}_{data_params(gene_feat_name)}_{data_params(cell_feat_name)}.pt"
        )
        heterodata_obj = torch.load(filepath)
        print(f"Loaded heterodata object from {filepath}")

    except:
        print("No file found, creating new one")
        graph_creator = Create_heterogeneous_graph(
            BASE_PATH=data_params(BASE_PATH),
            cancer_type=data_params(cancer_type),
            cell_feature=data_params(cell_feat_name),
            gene_feature=data_params(gene_feat_name),
            metapaths=graph_params(metapaths)
        )
        heterodata_obj = graph_creator.run_pipeline()

    