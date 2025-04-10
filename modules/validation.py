import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Dict, Tuple, List, Any
from modules.utils import construct_complete_predMatrix

def validate_model(
    model: torch.nn.Module,
    val_data: Any,
    device: str,
    loss_fn: torch.nn.Module,
    edge_type_label: str = "gene,dependency_of,cell"
) -> Tuple[float, float, float]:
    """
    Validate model performance on validation data.
    
    Args:
        model: The GNN model
        val_data: Validation data
        device: Device to run validation on ('cpu' or 'cuda:x')
        loss_fn: Loss function
        edge_type_label: Edge type label to use for prediction
        
    Returns:
        Tuple of (validation loss, AUC, AP)
    """
    model.eval()
    with torch.no_grad():
        val_data = val_data.to(device)
        out = model(val_data, edge_type_label=edge_type_label)
        
        pred = torch.sigmoid(out)
        ground_truth = val_data["gene", "dependency_of", "cell"].edge_label
        val_loss = loss_fn(out, ground_truth)
        
        # Check if there are positive examples in ground truth
        ground_truth_cpu = ground_truth.cpu()
        pred_cpu = pred.cpu()
        
        # For AP, we need at least one positive example
        if ground_truth_cpu.sum() > 0:
            ap_val = average_precision_score(ground_truth_cpu, pred_cpu)
        else:
            print("Warning: No positive examples in validation set, setting AP to 0.0")
            ap_val = 0.0
            
        # For AUC, we need at least one positive and one negative example
        if ground_truth_cpu.sum() > 0 and len(ground_truth_cpu) - ground_truth_cpu.sum() > 0:
            auc_val = roc_auc_score(ground_truth_cpu, pred_cpu)
        else:
            print("Warning: Not enough positive/negative examples in validation set for AUC calculation, setting AUC to 0.5")
            auc_val = 0.5
        
    return val_loss.item(), auc_val, ap_val

def evaluate_full_predictions(
    model: torch.nn.Module,
    full_pred_data: Any,
    cl_probs: torch.Tensor,
    cls_int: torch.Tensor,
    dep_genes: List[int],
    crispr_neurobl_continuous: pd.DataFrame,
    crispr_neurobl_bin: pd.DataFrame,
    device: str,
    edge_type_label: str = "gene,dependency_of,cell"
) -> Tuple[np.ndarray, float, List[float], List[float]]:
    """
    Evaluate full predictions on all genes and cell lines.
    
    Args:
        model: The GNN model
        full_pred_data: Data with all possible gene-cell pairs
        cl_probs: Tensor with all gene-cell pairs
        cls_int: Cell line indices
        dep_genes: List of dependency gene indices
        crispr_neurobl_continuous: CRISPR data with continuous values
        crispr_neurobl_bin: CRISPR data with binary values
        device: Device to run evaluation on ('cpu' or 'cuda:x')
        edge_type_label: Edge type label to use for prediction
        
    Returns:
        Tuple of (total predictions, average assay correlation, assay AP scores, gene AP scores)
    """
    model.eval()
    with torch.no_grad():
        full_pred_data = full_pred_data.to(device)
        total_preds = model(data=full_pred_data, edge_type_label=edge_type_label)
        total_preds_out = torch.sigmoid(total_preds).cpu().numpy()
        
        tot_pred_deps = construct_complete_predMatrix(
            total_predictions=total_preds_out,
            edge_index=cl_probs,
            index=cls_int.numpy(),
            columns=dep_genes
        )
        
        # Calculate correlation with CRISPR data (multiplied by -1 to reverse the direction of dependency)
        assay_corr = tot_pred_deps.corrwith(crispr_neurobl_continuous*-1, method='spearman', axis=1)
        
        # Calculate AP scores for each cell line and gene
        assay_ap, gene_ap = [], []
        
        # For each cell line calculate AP score
        valid_assay_count = 0
        for i, row in tot_pred_deps.iterrows():
            y_true = crispr_neurobl_bin.loc[i].values
            # Only calculate AP score if there is at least one positive example
            if np.sum(y_true) > 0:
                assay_ap.append(average_precision_score(
                    y_true=y_true,
                    y_score=row.values
                ))
                valid_assay_count += 1
        
        # For each gene calculate AP score
        valid_gene_count = 0
        for col in tot_pred_deps.columns:
            y_true = crispr_neurobl_bin[col].values
            # Only calculate AP score if there is at least one positive example
            if np.sum(y_true) > 0:
                gene_ap.append(average_precision_score(
                    y_true=y_true,
                    y_score=tot_pred_deps[col].values
                ))
                valid_gene_count += 1
        
        # Log the number of skipped calculations
        if len(tot_pred_deps.index) > valid_assay_count:
            print(f"Skipped AP calculation for {len(tot_pred_deps.index) - valid_assay_count} cell lines with no positive examples")
        if len(tot_pred_deps.columns) > valid_gene_count:
            print(f"Skipped AP calculation for {len(tot_pred_deps.columns) - valid_gene_count} genes with no positive examples")
            
    return total_preds_out, assay_corr.mean(), assay_ap, gene_ap 