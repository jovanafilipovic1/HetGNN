import torch
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple, Any
from copy import deepcopy
from torch_geometric.data import HeteroData
from torch_geometric.loader.link_neighbor_loader import LinkNeighborLoader
import torch_geometric.transforms as T
from models.HetGNN_Model_Jovana import HeteroData_GNNmodel
from modules.validation import validate_model, evaluate_full_predictions

def parse_model_parameters(
    model_params: Dict[str, Any],
    node_types: List[str],
    features_dim: Dict[str, int]
) -> Dict[str, Any]:
    """
    Parse model parameters from the configuration.
    
    Args:
        model_params: Model parameters from configuration
        node_types: List of node types
        features_dim: Dictionary of feature dimensions for each node type
        
    Returns:
        Dictionary of parsed parameters for model creation
    """
    # Parse hidden features
    hidden_features_str = model_params.get("hidden_features", "-1,256,128")
    hidden_features = [int(x) if x != '-1' else -1 for x in hidden_features_str.split(',')]
    
    # Handle the special case for features[0]
    if hidden_features[0] == -1:
        hidden_features[0] = (-1, -1)
    
    # Parse heads
    heads_str = model_params.get("heads", "1,1")
    heads = [int(x) for x in heads_str.split(',')]
    
    # Parse embedding dimensions
    emb_dim = str(model_params.get("emb_dim", 512))
    if ',' in emb_dim:
        emb_dims = emb_dim.split(',')
        emb_dim = {nt: int(ed) for nt, ed in zip(node_types, emb_dims)}
    else:
        emb_dim = int(emb_dim)
    
    return {
        "embedding_dim": emb_dim,
        "features": hidden_features,
        "heads": heads,
        "dropout": model_params.get("dropout", 0.2),
        "lp_model": model_params.get("lp_model", "simple"),
        "features_dim": features_dim,
        "aggregate": model_params.get("aggregate", "mean"),
    }

def prepare_model(
    heterodata_obj: HeteroData,
    config: Dict[str, Any],
    device: str
) -> HeteroData_GNNmodel:
    """
    Prepare the HetGNN model.
    
    Args:
        heterodata_obj: The heterogeneous graph data object
        config: Configuration dictionary containing model parameters
        device: Device to place the model on
        
    Returns:
        Initialized HetGNN model
    """
    # Extract parameters from config
    model_params = config['model_parameters']
    
    # Define node types
    node_types = ['gene', 'cell']
    
    # Define feature dimensions for each node type
    features_dim = {
        'gene': heterodata_obj['gene'].x.shape[1],
        'cell': heterodata_obj['cell'].x.shape[1]
    }
    
    # Parse model parameters
    parsed_params = parse_model_parameters(model_params, node_types, features_dim)
    
    # Create the model
    model = HeteroData_GNNmodel(
        heterodata=heterodata_obj,
        node_types=node_types,
        node_types_to_pred=node_types,
        embedding_dim=parsed_params["embedding_dim"],
        features=parsed_params["features"],
        heads=parsed_params["heads"],
        dropout=parsed_params["dropout"],
        act_fn=torch.nn.ReLU,
        lp_model=parsed_params["lp_model"],
        features_dim=parsed_params["features_dim"],
        aggregate=parsed_params["aggregate"],
        return_attention_weights=False
    )
    
    model.to(device)
    print(model)
    return model

def prepare_data_for_training(
    heterodata_obj: HeteroData,
    val_ratio: float,
    test_ratio: float,
    disjoint_train_ratio: float,
    batch_size: int
) -> Tuple[HeteroData, HeteroData, HeteroData, LinkNeighborLoader]:
    """
    Prepare data for training by splitting and creating loaders.
    
    Args:
        heterodata_obj: The heterogeneous graph data object
        val_ratio: Validation ratio
        test_ratio: Test ratio
        disjoint_train_ratio: Ratio of disjoint training edges
        batch_size: Batch size for training
        
    Returns:
        Tuple of (train_data, val_data, test_data, train_loader)
    """
    # Split graph in train/validation/test
    transform_traintest = T.RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        disjoint_train_ratio=disjoint_train_ratio,
        neg_sampling_ratio=0.0,  # No random negative sampling
        add_negative_train_samples=False,  # Don't add random negative samples
        edge_types=('gene', 'dependency_of', 'cell'),
        rev_edge_types=('cell', 'rev_dependency_of', 'gene'),
        is_undirected=True
    )

    train_data, val_data, test_data = transform_traintest(heterodata_obj)
    
    # Create the loader using the existing edge labels, without random negative sampling
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors={et: [-1]*2 for et in heterodata_obj.edge_types}, 
        edge_label_index=(
            ("gene", "dependency_of", "cell"),
            train_data["gene", "dependency_of", "cell"].edge_label_index
        ),
        edge_label=train_data["gene", "dependency_of", "cell"].edge_label,
        batch_size=batch_size,
        directed=True,
        shuffle=True,
        num_workers=1
    )
    
    return train_data, val_data, test_data, train_loader

def prepare_prediction_data(
    heterodata_obj: HeteroData,
    cls_int: torch.Tensor,
    dep_genes: List[int]
) -> Tuple[HeteroData, torch.Tensor]:
    """
    Prepare data for full prediction.
    
    Args:
        heterodata_obj: The heterogeneous graph data object
        cls_int: Cell line indices
        dep_genes: List of dependency gene indices
        
    Returns:
        Tuple of (full_pred_data, cl_probs)
    """
    # Define the full probability matrix for validation
    cl_probs = torch.zeros((2, len(cls_int)*len(dep_genes)), dtype=torch.long)

    for i, cl in enumerate(cls_int):
        x_ = torch.stack((
            torch.tensor(dep_genes), 
            torch.tensor([cl]*len(dep_genes))
        ), dim=0)
                        
        cl_probs[:, i*len(dep_genes):(i+1)*len(dep_genes)] = x_
        
    full_pred_data = heterodata_obj.clone()
    full_pred_data['gene', 'dependency_of', 'cell'].edge_label_index = cl_probs
    
    return full_pred_data, cl_probs

def train_model(
    config: Dict[str, Any],
    heterodata_obj: HeteroData,
    crispr_neurobl_continuous: pd.DataFrame,
    crispr_neurobl_bin: pd.DataFrame,
    cls_int: torch.Tensor,
    dep_genes: List[int],
    cells: List[str],
    genes: List[str],
    device: str,
    output_path: str
) -> Tuple[str, Dict[str, Any]]:
    """
    Train the HetGNN model.
    
    Args:
        config: Configuration dictionary
        heterodata_obj: The heterogeneous graph data object
        crispr_neurobl_continuous: CRISPR data with continuous values
        crispr_neurobl_bin: CRISPR data with binary values
        cls_int: Cell line indices
        dep_genes: List of dependency gene indices
        cells: List of cell names
        genes: List of gene names
        device: Device to train on
        output_path: Path to save output files
        
    Returns:
        Tuple of (model path, results dictionary)
    """
    # Extract parameters from config
    graph_params = config['graph_parameters']
    model_params = config['model_parameters']
    training_params = config['training_parameters']
    
    # Prepare data for training
    train_data, val_data, test_data, train_loader = prepare_data_for_training(
        heterodata_obj=heterodata_obj,
        val_ratio=training_params.get("validation_ratio", 0.1),
        test_ratio=training_params.get("test_ratio", 0.2),
        disjoint_train_ratio=training_params.get("disjoint_train_ratio", 0.0),
        batch_size=training_params.get("batch_size", 128)
    )
    
    # Create the model using prepare_model to avoid parameter parsing duplication
    hetGNNmodel = prepare_model(
        heterodata_obj=heterodata_obj,
        config=config,
        device=device
    )
    
    # Prepare data for full prediction
    full_pred_data, cl_probs = prepare_prediction_data(
        heterodata_obj=heterodata_obj,
        cls_int=cls_int,
        dep_genes=dep_genes
    )
    
    # Define training parameters
    optimizer = torch.optim.Adam(hetGNNmodel.parameters(), lr=model_params.get("lr", 0.01))
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    # Training state variables
    best_ap = 0
    best_ap_model = None
    best_epoch = 0
    n_epochs = model_params.get("epochs", 30)
    
    # Training loop
    assay_ap_total, gene_ap_total = [], []
    
    for epoch in range(n_epochs):
        # Training phase
        total_train_loss = 0
        hetGNNmodel.train()
        
        for sampled_data in train_loader:
            optimizer.zero_grad()
            sampled_data = sampled_data.to(device)
            
            if model_params.get("gcn_model", "simple") == 'gat':
                out = hetGNNmodel(sampled_data, edge_type_label="gene,dependency_of,cell")
            else:
                out = hetGNNmodel(sampled_data, edge_type_label="gene,dependency_of,cell")
                
            ground_truth = sampled_data["gene", "dependency_of", "cell"].edge_label
            loss = loss_fn(out, ground_truth)
            total_train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        # Validation phase
        ap_val, auc_val = 0, 0
        val_loss = 0
        
        if training_params.get("validation_ratio", 0.1) != 0.0:
            val_loss, auc_val, ap_val = validate_model(
                model=hetGNNmodel,
                val_data=val_data,
                device=device,
                loss_fn=loss_fn,
                edge_type_label="gene,dependency_of,cell",
                gcn_model=model_params.get("gcn_model", "simple")
            )
            
            if ap_val > best_ap:
                best_ap = ap_val
                best_ap_model = deepcopy(hetGNNmodel.state_dict())
                best_epoch = epoch
        
        # Evaluate on full prediction data
        total_preds_out, assay_corr_mean, assay_ap, gene_ap = evaluate_full_predictions(
            model=hetGNNmodel,
            full_pred_data=full_pred_data,
            cl_probs=cl_probs,
            cls_int=cls_int,
            dep_genes=dep_genes,
            crispr_neurobl_continuous=crispr_neurobl_continuous,
            crispr_neurobl_bin=crispr_neurobl_bin,
            device=device,
            edge_type_label="gene,dependency_of,cell"
        )
        
        assay_ap_total.append(assay_ap)
        gene_ap_total.append(gene_ap)
        
        # Print metrics
        print({
            'epoch': epoch, 
            'train loss': total_train_loss/len(train_loader),
            'val loss': val_loss if training_params.get("validation_ratio", 0.1) != 0.0 else "N/A", 
            'val auc': auc_val if training_params.get("validation_ratio", 0.1) != 0.0 else "N/A", 
            'val ap': ap_val if training_params.get("validation_ratio", 0.1) != 0.0 else "N/A",
            'assay_ap': np.mean(assay_ap), 
            'gene_ap': np.mean(gene_ap),
            'assay_corr_sp': assay_corr_mean
        })
    
    # Save the model
    os.makedirs(os.path.join(output_path, 'model'), exist_ok=True)
    cancer_type = graph_params.get("cancer_type", "All").replace(' ', '_')
    gene_feat = graph_params.get("gene_feat_name", "cgp")
    cell_feat = graph_params.get("cell_feat_name", "cnv")
    seed = config.get("settings", {}).get("seed", 42)
    
    model_path = os.path.join(
        output_path, 
        'model', 
        f'{gene_feat}-{cell_feat}-seedwith{seed}-at{best_epoch}.pt'
    )
    
    torch.save(best_ap_model, model_path)
    
    # Return the model path and training results
    results = {
        "model_path": model_path,
        "best_ap": best_ap,
        "best_epoch": best_epoch,
        "assay_ap_total": assay_ap_total,
        "gene_ap_total": gene_ap_total
    }
    
    return model_path, results 