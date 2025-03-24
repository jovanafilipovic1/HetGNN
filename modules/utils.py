import numpy as np
import pandas as pd
import torch
import os
from typing import List, Dict, Any, Tuple
from torch_geometric.data import HeteroData

def construct_complete_predMatrix(total_predictions: np.array,
                                  edge_index: torch.Tensor,
                                  index: list, 
                                  columns: list) -> pd.DataFrame:
    """
    Constructs a complete prediction matrix from model predictions.
    
    Args:
        total_predictions: Array of model predictions
        edge_index: Edge indices tensor
        index: List of indices for the dataframe (cell lines)
        columns: List of columns for the dataframe (genes)
        
    Returns:
        DataFrame with predicted dependencies for each cell line (row) and gene (column)
    """
    total_preds_df = pd.DataFrame({"gene": edge_index[0], "cell": edge_index[1], "prob": total_predictions})
    
    dep_df = pd.DataFrame(index=index, columns=columns, dtype=float) 
    for i in range(dep_df.shape[0]):
        tmp = total_preds_df.iloc[i*dep_df.shape[1]:(i+1)*dep_df.shape[1]]
        dep_df.loc[tmp.cell.iloc[0], tmp.gene.values] = tmp['prob'].values

    return dep_df

def process_data_and_create_mappings(
    heterodata_obj: HeteroData,
    config: Dict[str, Any],
    base_path: str
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, int],
    Dict[str, int],
    List[int],
    torch.Tensor,
    List[str],
    List[str]
]:
    """
    Process the data and create necessary mappings for the model.
    
    Args:
        heterodata_obj: The heterogeneous graph data object
        config: Configuration dictionary
        base_path: Base path for data files
        
    Returns:
        Tuple of processed data and mappings:
        - crispr_neurobl_continuous: CRISPR data with continuous values
        - crispr_neurobl_bin: CRISPR data with binary values
        - cell2int: Mapping from cell names to indices
        - gene2int: Mapping from gene names to indices
        - common_dep_genes: List of common dependency genes
        - cls_int: Cell line indices tensor
        - cells: List of cell names
        - genes: List of gene names
    """
    
    # Save cell and gene names 
    cells = heterodata_obj['cell'].names
    genes = heterodata_obj['gene'].names
    
    # Create mappings from names to indices
    cell2int = dict(zip(heterodata_obj['cell'].names, heterodata_obj['cell'].node_id.numpy()))
    gene2int = dict(zip(heterodata_obj['gene'].names, heterodata_obj['gene'].node_id.numpy()))
    
    # Get all genes that have a dependency edge
    dep_genes = list(set(heterodata_obj['gene', 'dependency_of', 'cell'].edge_index[0].numpy()))
    
    # Load and process CRISPR data
    crispr_neurobl = pd.read_csv(os.path.join(base_path, "Depmap/CRISPRGeneDependency.csv"), index_col=0)
    crispr_neurobl.columns = [col.split(' (')[0] for col in crispr_neurobl.columns]
    
    # Find intersection between CRISPR data and heterodata
    valid_cells = set(crispr_neurobl.index) & set(cells)
    valid_genes = set(crispr_neurobl.columns) & set(genes)
    
    # Filter CRISPR dataframe to only include common cells and genes
    filtered_crispr_df = crispr_neurobl.loc[list(valid_cells), list(valid_genes)]
    print(filtered_crispr_df.shape)
    
    # Create a copy for integer-indexed dataframe
    crispr_neurobl_continuous = filtered_crispr_df.copy(deep=True)
    
    # Convert indices and columns to integers
    crispr_neurobl_continuous.index = [cell2int[i] for i in filtered_crispr_df.index]
    crispr_neurobl_continuous.columns = [gene2int[i] for i in filtered_crispr_df.columns]
    
    # Further filter to only keep genes that have a dependency edge
    common_dep_genes = list(set(dep_genes) & set(crispr_neurobl_continuous.columns))
    crispr_neurobl_continuous = crispr_neurobl_continuous.loc[:, common_dep_genes]
    
    # Binarize CRISPR data
    crp_pos = config.get('graph_parameters', {}).get('crp_pos', 0.5)
    crispr_neurobl_bin = crispr_neurobl_continuous.applymap(lambda x: int(x > crp_pos))
    
    # Delete names attributes from heterodata object
    if hasattr(heterodata_obj['gene'], 'names'):
        del heterodata_obj['gene'].names
    if hasattr(heterodata_obj['cell'], 'names'):
        del heterodata_obj['cell'].names
    
    # Get cell line indices tensor
    cls_int = heterodata_obj['cell'].node_id
    
    return (
        crispr_neurobl_continuous,
        crispr_neurobl_bin,
        cell2int,
        gene2int,
        common_dep_genes,
        cls_int,
        cells,
        genes,
    ) 