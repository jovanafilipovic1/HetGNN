import torch
import os
import json
import itertools
import random
import time
import pandas as pd
import numpy as np
from copy import deepcopy
from typing import Dict, List, Tuple, Any

from modules.train import train_model, prepare_data_for_training, prepare_model
from modules.test import test_model, generate_full_predictions, save_results
from modules.validation import validate_model, evaluate_full_predictions

def generate_grid_combinations(hyperparam_grid: Dict[str, List]) -> List[Dict[str, Any]]:
    """
    Generate all combinations of hyperparameters from the grid.
    
    Args:
        hyperparam_grid: Dictionary of hyperparameter names to list of values
        
    Returns:
        List of dictionaries, each containing a specific combination of hyperparameters
    """
    keys, values = zip(*hyperparam_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # Shuffle combinations for better distribution
    random.seed(42)
    random.shuffle(combinations)
    
    return combinations

def update_config_with_hyperparams(config: Dict[str, Any], hyperparams: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration with specific hyperparameters.
    
    Args:
        config: Original configuration dictionary
        hyperparams: Hyperparameters to update
        
    Returns:
        Updated configuration dictionary
    """
    new_config = deepcopy(config)
    
    # Map hyperparameters to their respective sections in the config
    param_mapping = {
        "learning_rate": ("model_parameters", "lr"),
        "batch_size": ("training_parameters", "batch_size"),
        "hidden_features": ("model_parameters", "hidden_features"),
        "max_grad_norm": ("training_parameters", "max_grad_norm")
    }
    
    for param_name, param_value in hyperparams.items():
        section, key = param_mapping.get(param_name, (None, None))
        if section and key:
            new_config[section][key] = param_value
    
    return new_config

def run_grid_search(
    config: Dict[str, Any],
    heterodata_obj: Any,
    crispr_neurobl_continuous: pd.DataFrame,
    crispr_neurobl_bin: pd.DataFrame,
    cls_int: List[int],
    dep_genes: List[int],
    cells: List[str],
    genes: List[str],
    device: str,
    output_path: str
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Run grid search for hyperparameter tuning.
    
    Args:
        config: Base configuration dictionary
        heterodata_obj: Heterogeneous graph data object
        crispr_neurobl_continuous: CRISPR data with continuous values
        crispr_neurobl_bin: CRISPR data with binary values
        cls_int: Cell line indices
        dep_genes: List of dependency gene indices
        cells: List of cell names
        genes: List of gene names
        device: Device to train on
        output_path: Path to save output files
        
    Returns:
        Tuple of (best hyperparameters, best model results, best performance metrics)
    """
    if "hyperparam_grid" not in config:
        raise ValueError("No hyperparameter grid defined in the configuration")
    
    hyperparam_grid = config["hyperparam_grid"]
    combinations = generate_grid_combinations(hyperparam_grid)
    
    print(f"Running grid search with {len(combinations)} hyperparameter combinations")
    
    # Create a directory for grid search results
    grid_search_dir = os.path.join(output_path, 'grid_search')
    os.makedirs(grid_search_dir, exist_ok=True)

    # Track best model and its performance
    best_val_ap = 0.0
    best_hyperparams = None
    best_model = None
    best_results = None
    best_test_metrics = None
    
    # Save grid search configuration
    with open(os.path.join(grid_search_dir, 'grid_config.json'), 'w') as f:
        json.dump(hyperparam_grid, f, indent=2)
    
    # Results CSV file
    results_file = os.path.join(grid_search_dir, 'grid_results.csv')
    results_columns = [
        'learning_rate', 'batch_size', 'hidden_features', 'max_grad_norm',
        'val_ap', 'val_auc', 'train_loss', 'test_AP', 'test_gene_AP', 'test_assay_AP', 
        'training_time', 'epochs'
    ]
    
    # Create or append to results CSV
    if not os.path.exists(results_file):
        results_df = pd.DataFrame(columns=results_columns)
        results_df.to_csv(results_file, index=False)
    
    # Temporary directory for model checkpoints during grid search
    temp_model_dir = os.path.join(grid_search_dir, 'temp_models')
    os.makedirs(temp_model_dir, exist_ok=True)
    
    for idx, hyperparams in enumerate(combinations):
        print(f"\n[{idx+1}/{len(combinations)}] Running with hyperparameters: {hyperparams}")
        
        # Update configuration with this set of hyperparameters
        current_config = update_config_with_hyperparams(config, hyperparams)
        
        # Track timing
        start_time = time.time()
        
        try:
            # Train the model with these hyperparameters
            # Use the temp directory for intermediate models
            model_path, train_results = train_model(
                config=current_config,
                heterodata_obj=heterodata_obj,
                crispr_neurobl_continuous=crispr_neurobl_continuous,
                crispr_neurobl_bin=crispr_neurobl_bin,
                cls_int=cls_int,
                dep_genes=dep_genes,
                cells=cells,
                genes=genes,
                device=device,
                output_path=temp_model_dir
            )
            
            training_time = time.time() - start_time
            
            # Extract validation metrics
            val_ap = train_results.get("best_ap", 0.0)
            val_auc = train_results.get("val_auc", 0.0)
            train_loss = train_results.get("train_loss", 0.0)
            best_epoch = train_results.get("best_epoch", 0)
            
            # Test the model if test ratio is greater than 0
            test_metrics = {}
            if current_config["training_parameters"]["test_ratio"] > 0.0:
                # Prepare data for training to get the test data
                train_data, val_data, test_data, train_loader, val_loader = prepare_data_for_training(
                    heterodata_obj=heterodata_obj,
                    val_ratio=current_config["training_parameters"]["validation_ratio"],
                    test_ratio=current_config["training_parameters"]["test_ratio"],
                    disjoint_train_ratio=current_config["training_parameters"]["disjoint_train_ratio"],
                    batch_size=current_config["training_parameters"]["batch_size"]
                )
                
                # Load the best model
                model = prepare_model(
                    heterodata_obj=heterodata_obj,
                    config=current_config,
                    device=device
                )
                
                model.load_state_dict(torch.load(model_path))
                model.to(device)
                
                # Test the model
                test_metrics = test_model(
                    model=model,
                    test_data=test_data,
                    device=device,
                    edge_type_label="gene,dependency_of,cell"
                )
            
            # Store results
            result = {
                **hyperparams,
                'val_ap': val_ap,
                'val_auc': val_auc,
                'train_loss': train_loss,
                'test_AP': test_metrics.get("test_AP", 0.0),
                'test_gene_AP': test_metrics.get("test_gene_AP", 0.0),
                'test_assay_AP': test_metrics.get("test_assay_AP", 0.0),
                'training_time': training_time,
                'epochs': best_epoch
            }
            
            # Append to results CSV
            with open(results_file, 'a') as f:
                row = [str(result.get(col, '')) for col in results_columns]
                f.write(','.join(row) + '\n')
            
            # Check if this is the best model
            if val_ap > best_val_ap:
                best_val_ap = val_ap
                best_hyperparams = hyperparams
                best_results = train_results
                best_test_metrics = test_metrics
                
                # Save the model state dict
                best_model = torch.load(model_path)
                
                # Save best hyperparameters so far
                with open(os.path.join(grid_search_dir, 'best_hyperparams.json'), 'w') as f:
                    json.dump({
                        'hyperparams': best_hyperparams,
                        'val_ap': best_val_ap,
                        'val_auc': val_auc,
                        'train_loss': train_loss,
                        'test_metrics': best_test_metrics
                    }, f, indent=2)
            
        except Exception as e:
            print(f"Error during training with hyperparameters {hyperparams}: {str(e)}")
            # Log the error
            with open(os.path.join(grid_search_dir, 'errors.log'), 'a') as f:
                f.write(f"Hyperparameters: {hyperparams}, Error: {str(e)}\n")
    
    print(f"\nGrid search completed. Best validation AP: {best_val_ap}, with hyperparameters: {best_hyperparams}")
    
    # Save the best model in a standard location
    if best_model is not None and config["settings"]["save_full_predictions"]:
        # Create the best model config
        best_config = update_config_with_hyperparams(config, best_hyperparams)
        
        # Extract naming components
        cancer_type = best_config["graph_parameters"]["cancer_type"].replace(" ", "_") if best_config["graph_parameters"]["cancer_type"] else "All"
        gene_feat = best_config["graph_parameters"]["gene_feat_name"]
        cell_feat = best_config["graph_parameters"]["cell_feat_name"]
        model_type = best_config["model_parameters"].get("model_type", "gnn-gnn")
        seed = best_config["settings"]["seed"]
        best_epoch = best_results.get("best_epoch", 0)
        
        # Create model directory
        models_dir = os.path.join(output_path, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Save the best model with a standardized name
        best_model_path = os.path.join(
            models_dir, 
            f'{cancer_type}_{gene_feat}_{cell_feat}_{model_type}_seed{seed}_epoch{best_epoch}.pt'
        )
        
        torch.save(best_model, best_model_path)
        print(f"Saved best model to {best_model_path}")
        
        # Create a model for generating predictions
        model = prepare_model(
            heterodata_obj=heterodata_obj,
            config=best_config,
            device=device
        )
        
        model.load_state_dict(best_model)
        model.to(device)
        
        # Generate full predictions
        dfs, embs = generate_full_predictions(
            model=model,
            heterodata_obj=heterodata_obj,
            cls_int=cls_int,
            cells=cells,
            genes=genes,
            device=device,
            edge_type_label="gene,dependency_of,cell"
        )
        
        # Create best model output directory using the standard naming
        best_model_output_path = os.path.join(output_path, 'predictions')
        os.makedirs(best_model_output_path, exist_ok=True)
        
        # Save results
        save_results(
            config=best_config,
            dfs=dfs,
            output_path=best_model_output_path,
            seed=best_config["settings"]["seed"],
            crp_pos=best_config["graph_parameters"]["crp_pos"],
            plot_cell_embeddings=best_config["settings"]["plot_cell_embeddings"]
        )
    
    # Clean up temporary models
    import shutil
    try:
        shutil.rmtree(temp_model_dir)
        print(f"Cleaned up temporary model directory: {temp_model_dir}")
    except Exception as e:
        print(f"Warning: Could not clean up temporary models: {str(e)}")
    
    return best_hyperparams, best_results, best_test_metrics 