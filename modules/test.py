import torch
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple, Any
from sklearn.metrics import average_precision_score
from torch_geometric.data import HeteroData
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from modules.utils import construct_complete_predMatrix

def test_model(
    model: torch.nn.Module,
    test_data: HeteroData,
    device: str,
    edge_type_label: str = "gene,dependency_of,cell"
) -> Dict[str, float]:
    """
    Test the model on test data and return metrics.
    
    Args:
        model: The trained GNN model
        test_data: Test data
        device: Device to run testing on ('cpu' or 'cuda:x')
        edge_type_label: Edge type label to use for prediction
        
    Returns:
        Dictionary of test metrics
    """
    test_data = test_data.to(device)
    model.eval()
    
    with torch.no_grad():
        out = model(test_data, edge_type_label=edge_type_label)
        pred = torch.sigmoid(out).detach().cpu()
        ground_truth = test_data["gene", "dependency_of", "cell"].edge_label.detach().cpu()
        index = test_data["gene", "dependency_of", "cell"].edge_label_index.detach().cpu()
        
        # Calculate AP scores for each cell line
        test_assay_ap = []
        for cell in set(index[1].numpy()):
            assay_msk = index[1] == cell
            assay_msk = assay_msk.cpu()
            test_assay_ap.append(average_precision_score(
                y_true=ground_truth[assay_msk],
                y_score=pred[assay_msk]
            ))
           
        # Calculate AP scores for each gene
        test_gene_ap = []
        for gene in set(index[0].numpy()):
            gene_msk = index[0] == gene
            gene_msk = gene_msk.cpu()
            if ground_truth[gene_msk].sum() + pred[gene_msk].sum() > 0.5:
                test_gene_ap.append(average_precision_score(
                    y_true=ground_truth[gene_msk],
                    y_score=pred[gene_msk]
                ))
        
        # Calculate overall AP score
        ap_test = average_precision_score(ground_truth, pred)
        
    return {
        "test_AP": ap_test,
        "test_gene_AP": np.mean(test_gene_ap),
        "test_assay_AP": np.mean(test_assay_ap)
    }

def generate_full_predictions(
    model: torch.nn.Module,
    heterodata_obj: HeteroData,
    cls_int: torch.Tensor,
    cells: List[str],
    genes: List[str],
    device: str,
    edge_type_label: str = "gene,dependency_of,cell"
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, torch.Tensor]]:
    """
    Generate full predictions for all gene-cell pairs and extract embeddings.
    
    Args:
        model: The trained GNN model
        heterodata_obj: The heterogeneous graph data object
        cls_int: Cell line indices
        cells: List of cell names
        genes: List of gene names
        device: Device to run prediction on ('cpu' or 'cuda:x')
        edge_type_label: Edge type label to use for prediction
        
    Returns:
        Tuple of (dataframes dict, embeddings dict)
    """
    # Create a matrix with all possible gene-cell combinations
    cl_probs = torch.zeros((2, len(cls_int) * heterodata_obj['gene'].num_nodes), dtype=torch.long)
    
    for i, cl in enumerate(cls_int):
        x_ = torch.stack((
            heterodata_obj['gene'].node_id,
            torch.tensor([cl] * heterodata_obj['gene'].num_nodes)
        ), dim=0)
        cl_probs[:, i * heterodata_obj['gene'].num_nodes:(i + 1) * heterodata_obj['gene'].num_nodes] = x_
    
    # Prepare the data
    full_pred_data_all = heterodata_obj.clone()
    full_pred_data_all['gene', 'dependency_of', 'cell'].edge_label_index = cl_probs
    
    # Generate predictions and embeddings
    full_pred_data_all = full_pred_data_all.to(device)
    model.eval()
    
    with torch.no_grad():
        out_full_all, embs = model(
            data=full_pred_data_all, 
            edge_type_label=edge_type_label, 
            return_embeddings=True
        )
        preds_full_all = torch.sigmoid(out_full_all).cpu().numpy()
    
    # Create dataframes for embeddings
    gene_embs_df = pd.DataFrame(data=embs['gene'].cpu().detach().numpy(), index=genes)
    cell_embs_df = pd.DataFrame(data=embs['cell'].cpu().detach().numpy(), index=cells)
    
    # Create dataframes dict
    dfs = {
        "gene_embeddings": gene_embs_df,
        "cell_embeddings": cell_embs_df
    }
    
    # Construct prediction matrix
    tot_pred_deps = construct_complete_predMatrix(
        total_predictions=preds_full_all,
        edge_index=cl_probs,
        index=cls_int.numpy(),
        columns=heterodata_obj['gene'].node_id.numpy()
    )
    
    dfs["predictions"] = tot_pred_deps
    
    return dfs, embs

def save_results(
    config: Dict[str, Any],
    dfs: Dict[str, pd.DataFrame],
    output_path: str,
    seed: int,
    crp_pos: float = -1.5,
    plot_cell_embeddings: bool = False
) -> None:
    """
    Save results to files and potentially create visualizations.
    
    Args:
        config: Configuration dictionary
        dfs: Dictionary of dataframes with predictions and embeddings
        output_path: Path to save output files
        seed: Random seed used
        crp_pos: CRISPR threshold for positives
        plot_cell_embeddings: Whether to plot cell embeddings
    """
    # Extract parameters from config
    data_params = config['data']
    cancer_type = data_params.get("cancer_type", "All").replace(' ', '_')
    ppi = data_params.get("ppi", "Reactome")
    gene_feat = data_params.get("gene_feat_name", "cgp")
    cell_feat = data_params.get("cell_feat_name", "cnv")
    
    # Create output directories
    file_output_dir = os.path.join(output_path, 'file')
    figure_output_dir = os.path.join(output_path, 'Figures')
    os.makedirs(file_output_dir, exist_ok=True)
    os.makedirs(figure_output_dir, exist_ok=True)
    
    # Save gene embeddings
    gene_embs_file = os.path.join(
        file_output_dir,
        f"{cancer_type}_{ppi}_crispr{str(crp_pos).replace('.', '_')}_HetGNN_gene_embs_{gene_feat}_{cell_feat}_{seed}.csv"
    )
    dfs["gene_embeddings"].to_csv(gene_embs_file)
    print(f"Saved gene embeddings to {gene_embs_file}")
    
    # Save cell embeddings
    cell_embs_file = os.path.join(
        file_output_dir,
        f"{cancer_type}_{ppi}_crispr{str(crp_pos).replace('.', '_')}_HetGNN_cell_embs_{gene_feat}_{cell_feat}_{seed}.csv"
    )
    dfs["cell_embeddings"].to_csv(cell_embs_file)
    print(f"Saved cell embeddings to {cell_embs_file}")
    
    # Save predictions
    preds_file = os.path.join(
        file_output_dir,
        f"{cancer_type}_{ppi}_crispr{str(crp_pos).replace('.', '_')}_HetGNN_{gene_feat}_{cell_feat}_{seed}.csv"
    )
    dfs["predictions"].to_csv(preds_file)
    print(f"Saved predictions to {preds_file}")
    
    # Plot cell embeddings if requested
    if plot_cell_embeddings:
        plot_embeddings(
            dfs["cell_embeddings"],
            cells=dfs["cell_embeddings"].index,
            output_dir=figure_output_dir,
            base_path=data_params.get("base_path", "./Data/"),
            cancer_type=cancer_type,
            epoch=config.get("model", {}).get("epochs", 30)
        )

def plot_embeddings(
    cell_embs_df: pd.DataFrame,
    cells: List[str],
    output_dir: str,
    base_path: str,
    cancer_type: str,
    epoch: int
) -> None:
    """
    Plot cell embeddings using t-SNE.
    
    Args:
        cell_embs_df: DataFrame with cell embeddings
        cells: List of cell names
        output_dir: Directory to save the plot
        base_path: Base path for data files
        cancer_type: Cancer type for the title
        epoch: Number of epochs trained
    """
    # Apply t-SNE
    tsne = TSNE(n_components=2, verbose=1)
    tsne_results = tsne.fit_transform(cell_embs_df.values)
    tsne_results = pd.DataFrame(data=tsne_results, index=cell_embs_df.index, columns=["dim1", "dim2"])
    
    # Get cell metadata
    try:
        ccles = pd.read_csv(os.path.join(base_path, "Depmap/Model.csv"), header=0, index_col=0)
        if '_' in cancer_type:
            tsne_results['cancer_type'] = ccles.loc[cells, 'OncotreePrimaryDisease']
        else:
            tsne_results['cancer_type'] = ccles.loc[cells, 'OncotreeSubtype']
    except Exception as e:
        print(f"Could not load cell metadata: {e}")
        tsne_results['cancer_type'] = 'Unknown'
    
    # Create plot
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.scatterplot(
        x="dim1", y="dim2",
        hue="cancer_type",
        palette=sns.color_palette("colorblind", tsne_results['cancer_type'].unique().shape[0]),
        data=tsne_results,
        legend="full",
        alpha=0.8,
        ax=ax
    )
    plt.title(f"{cancer_type} clustering after {epoch} epochs")
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, f"{epoch}_GAT_tsne_embeddings.png"))
    plt.close()
    print(f"Saved t-SNE plot to {os.path.join(output_dir, f'{epoch}_GAT_tsne_embeddings.png')}") 