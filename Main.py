import torch
import os
import json
from typing import Dict, List, Tuple, Any, Optional, Union
from torch_geometric import seed_everything
from NetworkAnalysis import Create_heterogeneous_graph
from modules import (
    train_model, prepare_data_for_training, prepare_model,
    test_model, generate_full_predictions, save_results,
    process_data_and_create_mappings
)

if __name__ == "__main__":
    
    # Load configuration from the JSON file
    with open('config/parameters.json', 'r') as config_file:
        config = json.load(config_file)

    # Extract parameters with new structure
    settings = config["settings"]
    graph_params = config["graph_parameters"]
    model_params = config["model_parameters"]
    training_params = config["training_parameters"]
    
    # Print experiment name
    print("Experiment name:", settings["experiment_name"])
    
    # Set seed for reproducibility
    seed_everything(settings["seed"])

    # Check for GPU availability
    gpu_available = torch.cuda.is_available()
    print(f"GPU Available: {gpu_available}")
    if gpu_available:
        device = 'cuda:0'
    else:
        device = 'cpu'

    #read in (or create) heterodata_object
    try:
        filepath = os.path.join(
            graph_params["base_path"],
            'multigraphs',
            f'heteroData_gene_cell_{graph_params["cancer_type"].replace(" ", "_") if graph_params["cancer_type"] else "All"}_{graph_params["gene_feat_name"]}_{graph_params["cell_feat_name"]}_{"META" if graph_params["metapaths"] else ""}.pt'
        )
        heterodata_obj = torch.load(filepath)
        print(f"Loaded heterodata object from {filepath}")

    except Exception as e:
        print(f"No file found, creating new one: {e}")
        graph_creator = Create_heterogeneous_graph(
            BASE_PATH=graph_params["base_path"],
            cancer_type=graph_params["cancer_type"],
            cell_feature=graph_params["cell_feat_name"],
            gene_feature=graph_params["gene_feat_name"],
            metapaths=graph_params["metapaths"]
        )
        heterodata_obj = graph_creator.run_pipeline()
        print(heterodata_obj)
    
    # Process data and create mappings
    (
        crispr_neurobl_continuous,
        crispr_neurobl_bin,
        cell2int,
        gene2int,
        common_dep_genes,
        cls_int,
        cells,
        genes,
    ) = process_data_and_create_mappings(heterodata_obj, config, graph_params["base_path"])
    
    # Output results
    output_path = os.path.join(graph_params["base_path"], 'NB_results')
    os.makedirs(output_path, exist_ok=True)
    
    # Train the model
    model_path, train_results = train_model(
        config=config,
        heterodata_obj=heterodata_obj,
        crispr_neurobl_continuous=crispr_neurobl_continuous,
        crispr_neurobl_bin=crispr_neurobl_bin,
        cls_int=cls_int,
        dep_genes=common_dep_genes,
        cells=cells,
        genes=genes,
        device=device,
        output_path=output_path
    )
    
    # Test the model if test ratio is not 0
    if training_params["test_ratio"] > 0.0:
        # First, prepare data for training to get the test data
        train_data, val_data, test_data, _ = prepare_data_for_training(
            heterodata_obj=heterodata_obj,
            val_ratio=training_params["validation_ratio"],
            test_ratio=training_params["test_ratio"],
            disjoint_train_ratio=training_params["disjoint_train_ratio"],
            batch_size=training_params["batch_size"]
        )
        
        # Load the best model
        model = prepare_model(
            heterodata_obj=heterodata_obj,
            config=config,
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
        
        # Print test metrics
        print("Test metrics:", test_metrics)
    
    # Generate full predictions if requested
    if settings["save_full_predictions"]:
        # Load the best model
        model = prepare_model(
            heterodata_obj=heterodata_obj,
            config=config,
            device=device
        )
        
        model.load_state_dict(torch.load(model_path))
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
        
        # Save results
        save_results(
            config=config,
            dfs=dfs,
            output_path=output_path,
            seed=settings["seed"],
            crp_pos=graph_params["crp_pos"],
            plot_cell_embeddings=settings["plot_cell_embeddings"]
        )
        
    print("Training and testing completed successfully!") 