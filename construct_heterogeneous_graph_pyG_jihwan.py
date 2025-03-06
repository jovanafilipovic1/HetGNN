from NetworkAnalysis.UndirectedInteractionNetwork import UndirectedInteractionNetwork
import torch_geometric.transforms as T
from torch_geometric.transforms import to_undirected
from torch_geometric.data import HeteroData
import pandas as pd
import numpy as np
import gseapy as gp
import random
import pickle
import torch

cancer_type = 'Neuroblastoma'
#cancer_type = 'Non-Small Cell Lung Cancer_Lung Neuroendocrine Tumor'
BASE_PATH = "./Data/"
ppi = "Reactome"
remove_rpl = "_noRPL"
remove_commonE = ""
useSTD = "STD"
crispr_threshold_pos = -1.5
cell_feat_name = "cnv"
gene_feat_name = 'cgp'

with open(BASE_PATH+f"multigraphs/{cancer_type.replace(' ', '_')}_{ppi}{remove_rpl}_{useSTD}{remove_commonE}_crispr{str(crispr_threshold_pos).replace('.','_')}.pickle", 'rb') as handle:
    mg_obj = pickle.load(handle) #3columns: gene_A (gene or cell line), gene_B, type (scaffold or depmap)
#type2nodes and int2gene are both functions defined in NetworkAnalys
#all_genes_int is a list of integers that represent the genes 
#all_genes_name is a list of gene names
all_genes_int = mg_obj.type2nodes['gene'] 
all_genes_name = [mg_obj.int2gene[i] for i in all_genes_int]

# PPI obj
ppi_obj = mg_obj.getEdgeType_subset(edge_type='scaffold') #gene-gene interactions
ppi_obj_new_gene2int = {n:i for i, n in enumerate(all_genes_name)}
ppi_obj_new_int2gene = {v:k for k, v in ppi_obj_new_gene2int.items()}
ppi_interactions = ppi_obj.getInteractionNamed() #function defined in multigraph.py, it returns a dataframe with 2 columns "Gene_A", "Gene_B"
ppi_interactions = ppi_interactions.map(lambda x: ppi_obj_new_gene2int[x]) #replace gene names with their corresponding integers

# DEP obj
dep_obj = mg_obj.getEdgeType_subset(edge_type='depmap')
cells = [k for k, v in mg_obj.node_type_names.items() if v == 'cell']
cell2int = {c:i for i, c in enumerate(cells)}
int2cell = {v:k for k, v in cell2int.items()}
dep_interactions = dep_obj.getInteractionNamed() #2 columns: Gene_A (=cell line) and Gene_B (=gene)
dep_genes = [dep_obj.int2gene[i] for i in dep_obj.type2nodes['gene']] #list of genes with a dependecy edge

dep_interactions.loc[~dep_interactions.Gene_A.isin(cells), ['Gene_A', 'Gene_B']] = \
    dep_interactions.loc[~dep_interactions.Gene_A.isin(cells), ['Gene_B', 'Gene_A']].values # assure that all values in Gene_A are cells, otherwise switch columns

assert (dep_interactions.Gene_A.isin(cells).sum() == dep_interactions.shape[0] , "This is not a depencency (two genes)") #all Gene_A should be cell lines
dep_interactions = dep_interactions.map(lambda x: cell2int[x] if x in cell2int else ppi_obj_new_gene2int[x]) #map cell lines and genes to their resp. integers
dep_interactions = dep_interactions[['Gene_B', 'Gene_A']] #switch columns (gene, cell line)
print(dep_interactions.shape)

# cell gene 합쳐서 int로 되어있어서 각각 분리

# Oversample low pos -------------------------------------------------------------------------------------------------------------------------------------
# crispr_neurobl = pd.read_csv(BASE_PATH+f"data/crispr_{cancer_type}_{ppi}.csv", index_col=0)
# crispr_neurobl_int = crispr_neurobl.copy(deep=True)
# crispr_neurobl_int.index = [cell2int[i] for i in crispr_neurobl.index]
# crispr_neurobl_int.columns = [ppi_obj_new_gene2int[i] for i in crispr_neurobl.columns]
# dep_genes = [ppi_obj_new_gene2int[i] for i in dep_obj.node_names if i not in cells]
# crispr_neurobl_int = crispr_neurobl_int.loc[:, dep_genes]
# crispr_neurobl_bin = crispr_neurobl_int.applymap(lambda x: int(x < crispr_threshold_pos))

# to_sample = len(cells)-crispr_neurobl_bin.sum()
# for gi, tosample in to_sample.iteritems():
#     possible_edges = list(map(tuple, dep_interactions[dep_interactions.Gene_B == gi].values))
#     to_concat = pd.DataFrame(random.choices(population=possible_edges, k=tosample), columns=['Gene_B', 'Gene_A'], dtype=int)
#     dep_interactions = pd.concat([dep_interactions, to_concat])
# print(dep_interactions.shape)

# -------------------------------------------------------------------------------------------------------------------------------------

def read_gmt_file(fp, nw_obj): #read gmt files: generates a dictionary of genes
    genes_per_DB = {}
    if isinstance(nw_obj, list):
        focus_genes = set(nw_obj)
    else:
        focus_genes = set(nw_obj.node_names)
    with open(fp) as f:
        lines = f.readlines()
        for line in lines:
            temp = line.strip('\n').split('\t')
            genes_per_DB[temp[0]] = set(gene for gene in temp[2:]) & focus_genes
    return genes_per_DB

# Gene features
if gene_feat_name == 'cgp':
    cgn = read_gmt_file(BASE_PATH+"MsigDB/c2.cgp.v2023.2.Hs.symbols.gmt", ppi_obj) #ppi_obj is used to filter out genes that are not in the network
elif gene_feat_name == 'bp':
    cgn = read_gmt_file(BASE_PATH+"MsigDB/c5.go.bp.v2023.2.Hs.symbols.gmt", ppi_obj)
elif gene_feat_name == 'go':    
    cgn = read_gmt_file(BASE_PATH+"MsigDB/c5.go.v2023.2.Hs.symbols.gmt", ppi_obj)
elif gene_feat_name == 'cp':  
    cgn = read_gmt_file(BASE_PATH+"MsigDB/c2.cp.v2023.2.Hs.symbols.gmt", ppi_obj)

# Create a dataframe with all genes (rows) and their corresponding gene sets (columns)
cgn_df = pd.DataFrame(np.zeros((len(all_genes_name), len(cgn))), index=all_genes_name, columns=list(cgn.keys()))
for k, v in cgn.items():
    cgn_df.loc[list(v), k] = 1  #set 1 if gene is in the gene set, 0 otherwise
zero_gene_feat = cgn_df.index[cgn_df.sum(axis=1) == 0] # This is not allowed because all genes must have features
# Check how many of the dep genes are in that all 0, otherwise this is basically of no use
zero_depgenes = set(zero_gene_feat) & set(dep_genes) #genes that are in the dep_genes and have no features
print(len(zero_depgenes)) #len = 1, but why not removing ??

#gene featur matrix (rows=genes) 
gene_feat = torch.from_numpy(cgn_df.values).to(torch.float) ##why not filtering???? 크키맞출라고

# Cell features
if cell_feat_name == "expression":
    path = BASE_PATH+'Depmap/OmicsExpressionProteinCodingGenesTPMLogp1.csv'
    ccle_expression = pd.read_csv(path, header=0, index_col=0)
    ccle_expression.columns = [i.split(' ')[0] for i in ccle_expression.columns]
    # subset_nodes = list(set(ccle_expression.columns) & set(all_genes_name))
    cancer_expression = ccle_expression.loc[list(set(cells) & set(ccle_expression.index))]

    hvg_q = cancer_expression.std().quantile(q=0.95) #threshold for high variance genes
    hvg_final = cancer_expression.std()[cancer_expression.std() >= hvg_q].index #select genes with high variance

    cancer_expression_hvg = cancer_expression[hvg_final]
    # cancer_expression_full = pd.concat([cancer_expression,
    #                                     pd.DataFrame(np.tile(cancer_expression.mean().values, (len(set(cells) - set(cancer_expression.index)), 1)),
    #                                                  index=list(set(cells) - set(cancer_expression.index)), columns=cancer_expression.columns)])

    #filling the missing entries with the mean expression values of the corresponding genes.
    cancer_expression_full = pd.concat([cancer_expression_hvg,
                                        pd.DataFrame(np.tile(cancer_expression_hvg.mean().values, (len(set(cells) - set(cancer_expression_hvg.index)), 1)),
                                                    index=list(set(cells) - set(cancer_expression_hvg.index)), columns=cancer_expression_hvg.columns)])
    #cancer_expression_full[:] = np.random.uniform(0, 10, size=cancer_expression_full.shape) #delete this line!!! It generates random values (between 0 and 10) for each cell 
    cell_feat = torch.from_numpy(cancer_expression_full.loc[cell2int.keys()].values).to(torch.float) #cell featture matrix with rows as cell lines and columns as genes with high variance

elif cell_feat_name == "cnv":
    path = BASE_PATH+'Depmap/OmicsCNGene.csv'
    ccle_cnv = pd.read_csv(path, header=0, index_col=0)
    ccle_cnv.columns = [i.split(' ')[0] for i in ccle_cnv.columns] 
    ccle_cnv = ccle_cnv[ccle_cnv.columns[ccle_cnv.isna().sum() == 0]] #remove columns with missing values
    ccle_cnv = ccle_cnv.loc[list(set(cells) & set(ccle_cnv.index))] #filter only the cells that are in the cell lines

    hvg_q = ccle_cnv.std().quantile(q=0.95)  #compute the 95th percentile of the standard deviation per gene accross all cells
    hvg_final = ccle_cnv.std()[ccle_cnv.std() >= hvg_q].index #select genes with high variance (top 5%)

    ccle_cnv_hvg = ccle_cnv[hvg_final]
    cell_feat = torch.from_numpy(ccle_cnv_hvg.loc[cell2int.keys()].values).to(torch.float) #cell feature matrix

elif cell_feat_name == "cnv_abs":
    path = BASE_PATH+'Depmap/OmicsAbsoluteCNGene.csv'
    ccle_cnv = pd.read_csv(path, header=0, index_col=0)
    ccle_cnv.columns = [i.split(' ')[0] for i in ccle_cnv.columns]
    ccle_cnv = ccle_cnv[ccle_cnv.columns[ccle_cnv.isna().sum() == 0]]
    ccle_cnv = ccle_cnv.loc[list(set(cells) & set(ccle_cnv.index))]

    hvg_q = ccle_cnv.std().quantile(q=0.95)
    hvg_final = ccle_cnv.std()[ccle_cnv.std() >= hvg_q].index

    ccle_cnv_hvg = ccle_cnv[hvg_final]
    ccle_cnv_hvg[:] = np.random.uniform(0, 10, size=ccle_cnv_hvg.shape)
    cell_feat = torch.from_numpy(ccle_cnv_hvg.loc[cell2int.keys()].values).to(torch.float)

elif '_' in cell_feat_name:
    all_feats = cell_feat_name.split('_')
    for feat in all_feats:
        if feat == "expression":
            path = BASE_PATH+'Depmap/OmicsExpressionProteinCodingGenesTPMLogp1.csv'
            ccle_expression = pd.read_csv(path, header=0, index_col=0)
            ccle_expression.columns = [i.split(' ')[0] for i in ccle_expression.columns]
            # subset_nodes = list(set(ccle_expression.columns) & set(all_genes_name))
            cancer_expression = ccle_expression.loc[list(set(cells) & set(ccle_expression.index))]

            hvg_q_expression = cancer_expression.std().quantile(q=0.95)
            hvg_final_expression = cancer_expression.std()[cancer_expression.std() >= hvg_q_expression].index

            cancer_expression_hvg = cancer_expression[hvg_final_expression]
            # cancer_expression_full = pd.concat([cancer_expression,
            #                                     pd.DataFrame(np.tile(cancer_expression.mean().values, (len(set(cells) - set(cancer_expression.index)), 1)),
            #                                                  index=list(set(cells) - set(cancer_expression.index)), columns=cancer_expression.columns)])
            cancer_expression_full = pd.concat([cancer_expression_hvg,
                                                pd.DataFrame(np.tile(cancer_expression_hvg.mean().values, (len(set(cells) - set(cancer_expression_hvg.index)), 1)),
                                                            index=list(set(cells) - set(cancer_expression_hvg.index)), columns=cancer_expression_hvg.columns)])

        elif feat == "cnv":
            path = BASE_PATH+'Depmap/OmicsCNGene.csv'
            ccle_cnv = pd.read_csv(path, header=0, index_col=0)
            ccle_cnv.columns = [i.split(' ')[0] for i in ccle_cnv.columns]
            ccle_cnv = ccle_cnv[ccle_cnv.columns[ccle_cnv.isna().sum() == 0]]
            ccle_cnv = ccle_cnv.loc[list(set(cells) & set(ccle_cnv.index))]

            hvg_q = ccle_cnv.std().quantile(q=0.95)
            hvg_final_CNV = ccle_cnv.std()[ccle_cnv.std() >= hvg_q].index

            ccle_cnv_hvg = ccle_cnv[hvg_final_CNV]

    if all_feats[0] == 'expression':
        expression_CNV_full = pd.concat([cancer_expression_full, ccle_cnv_hvg], axis=1)  
        cell_feat = torch.from_numpy(expression_CNV_full.loc[cell2int.keys()].values).to(torch.float) 
    else:
        CNV_expression_full = pd.concat([ccle_cnv_hvg, cancer_expression_full], axis=1)  
        cell_feat = torch.from_numpy(CNV_expression_full.loc[cell2int.keys()].values).to(torch.float)

    
elif "SM" in cell_feat_name:
    path = BASE_PATH+'Depmap/OmicsSomaticMutationsMatrixDamaging.csv'
    ccle_SM = pd.read_csv(path, header=0, index_col=0)
    ccle_SM.columns = [i.split(' ')[0] for i in ccle_SM.columns]
    ccle_SM = ccle_SM[ccle_SM.columns[ccle_SM.isna().sum() == 0]]
    ccle_SM = ccle_SM.loc[list(set(cells) & set(ccle_SM.index))]
    ccle_SM_final = ccle_SM.loc[:, (ccle_SM.sum() != 0)]

    missing_cells = set(cell2int.keys()) - set(ccle_SM_final.index) #remove columns (genes) where no cell line has a mutation

    for cell in missing_cells:
        ccle_SM_final.loc[cell] = np.zeros(ccle_SM_final.shape[1])
    cell_feat = torch.from_numpy(ccle_SM_final.loc[cell2int.keys()].values).to(torch.float)  
 

elif "MYCN" in cell_feat_name:
    ccle_MYCN = pd.read_csv('/data/jilim/HetGNN/MYCN_binary.csv', header=0, index_col=0)
    cell_feat = torch.from_numpy(ccle_MYCN.loc[cell2int.keys()].values).to(torch.float)    

# Construction of the PyTroch geometric heterogeneous graph
data = HeteroData()

# First construt the node ids, easy from the MG obj
data['gene'].node_id = torch.tensor(list(ppi_obj_new_gene2int.values())) #gene node ids: integers
data['gene'].names = list(ppi_obj_new_gene2int.keys()) #gene names
data['cell'].node_id = torch.tensor(list(cell2int.values()))
data['cell'].names = list(cell2int.keys())

# Add the node features and edge indices
data['gene'].x = gene_feat
data['cell'].x = cell_feat

data['gene', 'interacts_with', 'gene'].edge_index = torch.tensor(ppi_interactions.values.transpose(), dtype=torch.long)
data['gene', 'dependency_of', 'cell'].edge_index = torch.tensor(dep_interactions.values.transpose(), dtype=torch.long)

# Convert to undirected graph
data = T.ToUndirected(merge=False)(data)
assert data.validate()

print(data)

torch.save(obj=data, f=BASE_PATH+f"multigraphs/heteroData_gene_cell_{cancer_type.replace(' ', '_')}_{ppi}"\
          f"_crispr{str(crispr_threshold_pos).replace('.','_')}_{gene_feat_name}_{cell_feat_name}.pt")