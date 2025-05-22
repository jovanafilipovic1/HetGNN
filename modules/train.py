import torch
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple, Any, Optional
from copy import deepcopy
from torch_geometric.data import HeteroData
from torch_geometric.loader.link_neighbor_loader import LinkNeighborLoader
import torch_geometric.transforms as T
from models.HetGNN_Model_Jovana import HeteroData_GNNmodel_Jovana
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
    hidden_features = []
    
    # Process each value in the hidden_features string
    for x in hidden_features_str.split(','):
        if x == '-1':
            hidden_features.append(-1)
        else:
            hidden_features.append(int(x))
    
    # Parse embedding dimensions
    emb_dim = str(model_params.get("emb_dim", 512))
    if ',' in emb_dim:
        emb_dims = emb_dim.split(',')
        emb_dim = {nt: int(ed) for nt, ed in zip(node_types, emb_dims)}
    else:
        emb_dim = int(emb_dim)
    
    # Parse activation function
    activation_function = model_params.get("activation_function", "sigmoid")
    if activation_function.lower() == "relu":
        act_fn = torch.nn.ReLU
    elif activation_function.lower() == "sigmoid":
        act_fn = torch.nn.Sigmoid
    elif activation_function.lower() == "tanh":
        act_fn = torch.nn.Tanh
    elif activation_function.lower() == "leakyrelu":
        act_fn = torch.nn.LeakyReLU
    else:
        # Default to ReLU if not recognized
        act_fn = torch.nn.ReLU
    
    return {
        "embedding_dim": emb_dim,
        "hidden_features": hidden_features,
        "dropout": model_params.get("dropout", 0.2),
        "lp_model": model_params.get("lp_model", "simple"),
        "features_dim": features_dim,
        "aggregate": model_params.get("aggregate", "mean"),
        "model_type": model_params.get("model_type", "gnn-gnn"),  # Default to gnn-gnn
        "act_fn": act_fn
    }

def prepare_model(
    heterodata_obj: HeteroData,
    config: Dict[str, Any],
    device: str ):
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
    model = HeteroData_GNNmodel_Jovana(
        heterodata=heterodata_obj,
        node_types=node_types,
        node_types_to_pred=node_types,
        embedding_dim=parsed_params["embedding_dim"],
        dropout=parsed_params["dropout"],
        act_fn=parsed_params["act_fn"],
        lp_model=parsed_params["lp_model"],
        features_dim=parsed_params["features_dim"],
        aggregate=parsed_params["aggregate"],
        model_type=parsed_params["model_type"],
        hidden_features=parsed_params["hidden_features"]
    )
        
    model.to(device)
    print(model)
    return model

def prepare_data_for_training(
    heterodata_obj: HeteroData,
    val_ratio: float,
    test_ratio: float,
    disjoint_train_ratio: float,
    batch_size: int,
    num_neighbors: List[int] = [-1, -1],
    cancer_type: str = None
) -> Tuple[HeteroData, HeteroData, HeteroData, LinkNeighborLoader, Optional[LinkNeighborLoader]]:
    """
    Prepare data for training by splitting and creating loaders.
    
    Args:
        heterodata_obj: The heterogeneous graph data object
        val_ratio: Validation ratio
        test_ratio: Test ratio
        disjoint_train_ratio: Ratio of disjoint training edges
        batch_size: Batch size for training
        num_neighbors: Number of neighbors to sample per layer, default samples all neighbors
        cancer_type: Type of cancer, if None will create validation loader
        
    Returns:
        Tuple of (train_data, val_data, test_data, train_loader, val_loader)
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
        is_undirected=False, 
    )

    train_data, val_data, test_data = transform_traintest(heterodata_obj)
    
    # Optimize number of workers based on CPU cores
    num_workers = min(4, os.cpu_count() or 1)  # Use up to 4 workers
    
    # Create the loader using the existing edge labels, without random negative sampling
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors={et: num_neighbors for et in heterodata_obj.edge_types}, 
        edge_label_index=(
            ("gene", "dependency_of", "cell"),
            train_data["gene", "dependency_of", "cell"].edge_label_index
        ),
        edge_label=train_data["gene", "dependency_of", "cell"].edge_label,
        batch_size=batch_size,
        directed=True,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,  # Keep workers alive between epochs
        pin_memory=True  # Pin memory for faster GPU transfer
    )
    
    # Create validation loader only if cancer_type is None
    val_loader = None
    if cancer_type is None and val_ratio > 0:
        val_loader = LinkNeighborLoader(
            data=val_data,
            num_neighbors={et: num_neighbors for et in heterodata_obj.edge_types},
            edge_label_index=(
                ("gene", "dependency_of", "cell"),
                val_data["gene", "dependency_of", "cell"].edge_label_index
            ),
            edge_label=val_data["gene", "dependency_of", "cell"].edge_label,
            batch_size=batch_size,
            directed=True,
            shuffle=False,  # No need to shuffle validation data
            num_workers=num_workers,
            persistent_workers=True,
            pin_memory=True
        )
    
    return train_data, val_data, test_data, train_loader, val_loader

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
    cls_int: List[int],
    dep_genes: List[int],
    cells: List[str],
    genes: List[str],
    device: str,
    output_path: str
) -> Tuple[str, Dict[str, Any]]:
    """
    Train the HetGNN model with early stopping based on validation loss.
    
    The training will stop when either:
    1. The maximum number of epochs (100) is reached
    2. The validation loss does not improve for 12 consecutive epochs
    
    The best model (based on validation AP score) will be saved.
    
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
    
    # Get num_neighbors parameter from config, handling both list and string formats
    num_neighbors_param = training_params.get("num_neighbors", [-1, -1])
    
    # Handle case where num_neighbors is a string (like "40,40")
    if isinstance(num_neighbors_param, str):
        num_neighbors = [int(x) for x in num_neighbors_param.split(',')]
    else:
        num_neighbors = num_neighbors_param
    
    # Prepare data for training
    train_data, val_data, test_data, train_loader, val_loader = prepare_data_for_training(
        heterodata_obj=heterodata_obj,
        val_ratio=training_params.get("validation_ratio", 0.1),
        test_ratio=training_params.get("test_ratio", 0.2),
        disjoint_train_ratio=training_params.get("disjoint_train_ratio", 0.0),
        batch_size=training_params.get("batch_size", 128),
        num_neighbors=num_neighbors,
        cancer_type=graph_params.get("cancer_type")
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
    optimizer = torch.optim.AdamW(  
        hetGNNmodel.parameters(), 
        lr=model_params.get("lr", 0.0001),
        weight_decay=0.01  # Add weight decay for regularization
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    # Training state variables
    best_ap = 0
    best_ap_model = None
    best_epoch = 0
    # Early stopping parameters
    patience = 12  # Stop training if no improvement for 12 epochs
    max_epochs = 100  # Maximum number of epochs to train
    best_val_loss = float('inf')
    counter = 0  # Count epochs without improvement
    
    # Get actual epochs setting from config with a max of 100
    n_epochs = min(model_params.get("max_epochs", max_epochs), max_epochs)
    
    # Training loop
    assay_ap_total, gene_ap_total = [], []
    
    # Get gradient clipping parameter with a default value of 1.0
    max_grad_norm = training_params.get("max_grad_norm", 1.0)
    print(f"Using gradient clipping with max_norm={max_grad_norm}")
    
    for epoch in range(n_epochs):
        
        # Training phase
        total_train_loss = 0
        hetGNNmodel.train()
        
        for sampled_data in train_loader:
            optimizer.zero_grad()
            sampled_data = sampled_data.to(device)
            
            out = hetGNNmodel(sampled_data, edge_type_label="gene,dependency_of,cell")
                
            ground_truth = sampled_data["gene", "dependency_of", "cell"].edge_label
            loss = loss_fn(out, ground_truth)
            total_train_loss += loss.item()
            
            loss.backward()
            
            # Apply gradient clipping to prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(hetGNNmodel.parameters(), max_norm=max_grad_norm)
            
            optimizer.step()
        
        # Validation phase
        ap_val, auc_val = 0, 0
        val_loss = 0
        
        if val_loader is not None:
            # Use validation loader if available
            val_loss, auc_val, ap_val = validate_model(
                model=hetGNNmodel,
                val_data=val_loader,
                device=device,
                loss_fn=loss_fn,
                edge_type_label="gene,dependency_of,cell"
            )
        elif training_params.get("validation_ratio", 0.1) != 0.0:
            # Fall back to old validation method if no loader
            val_loss, auc_val, ap_val = validate_model(
                model=hetGNNmodel,
                val_data=val_data,
                device=device,
                loss_fn=loss_fn,
                edge_type_label="gene,dependency_of,cell"
            )
            
        # Check for improvement in validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0  # Reset counter
        else:
            counter += 1  # Increment counter
        
        # Check for improvement in AP
        if ap_val > best_ap:
            best_ap = ap_val
            best_ap_model = deepcopy(hetGNNmodel.state_dict())
            best_epoch = epoch
            
        # Early stopping check
        if counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs. No improvement for {patience} consecutive epochs.")
            break
        
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
        
        # Only append metrics if there are valid values
        if len(assay_ap) > 0:
            assay_ap_total.append(np.mean(assay_ap))
        if len(gene_ap) > 0:
            gene_ap_total.append(np.mean(gene_ap))
        
        # Print metrics
        print({
            'epoch': epoch, 
            'train loss': total_train_loss/len(train_loader),
            'val loss': val_loss if training_params.get("validation_ratio", 0.1) != 0.0 else "N/A", 
            'val auc': auc_val if training_params.get("validation_ratio", 0.1) != 0.0 else "N/A", 
            'val ap': ap_val if training_params.get("validation_ratio", 0.1) != 0.0 else "N/A",
            'assay_ap': np.mean(assay_ap) if len(assay_ap) > 0 else "N/A", 
            'gene_ap': np.mean(gene_ap) if len(gene_ap) > 0 else "N/A",
            'assay_corr_sp': assay_corr_mean
        })
    
    # Save the model
    os.makedirs(os.path.join(output_path, 'model'), exist_ok=True)
    cancer_type = graph_params['cancer_type']
    gene_feat = graph_params.get("gene_feat_name", "cgp")
    cell_feat = graph_params.get("cell_feat_name", "cnv")
    seed = config.get("settings", {}).get("seed", 42)
    
    model_path = os.path.join(
        output_path, 
        'model', 
        f'{cancer_type.replace(" ", "_") if cancer_type else "All"}-{gene_feat}-{cell_feat}-seedwith{seed}-at{best_epoch}.pt'
    )
    
    torch.save(best_ap_model, model_path)
    
    # Return the model path and training results
    results = {
        "model_path": model_path,
        "best_ap": best_ap,
        "best_epoch": best_epoch,
        "val_auc": auc_val if training_params.get("validation_ratio", 0.1) != 0.0 else 0.0,
        "train_loss": total_train_loss/len(train_loader) if len(train_loader) > 0 else 0.0,
        "assay_ap_total": assay_ap_total,
        "gene_ap_total": gene_ap_total
    }
    
    return model_path, results 