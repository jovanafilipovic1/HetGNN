import sys
#sys.path.append('/Users/jovanafilipovic/Downloads/MSc Bioinformatics/Year 2/Thesis/Python_scripts')

from NetworkAnalysis.MultiGraph import MultiGraph
from NetworkAnalysis.UndirectedInteractionNetwork import UndirectedInteractionNetwork
from itertools import combinations

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import os

BASE_PATH = "./Data/"
cancer_type = "Neuroblastoma"
train_ratio = 0.8
ppi = "Reactome"
remove_rpl = "_noRPL"
remove_commonE = ""  #this will not be removed
useSTD = "STD"
crispr_threshold_pos = -1.5
ppi_train_ratio = 0.8

# Read in all relevant DepMap data
ccles_ori = pd.read_csv(BASE_PATH+"Depmap/Model.csv", index_col=0)
# ccles = ccles_ori.loc[ccles_ori.PatientID.drop_duplicates().index]
ccles = ccles_ori
ccles['OncotreePrimaryDisease'].value_counts()

path = BASE_PATH+'Depmap/CRISPRGeneEffect.csv'
crispr_effect = pd.read_csv(path, header=0, index_col=0)
crispr_effect.columns = [i.split(' ')[0] for i in crispr_effect.columns] #extract the gene name (first part of the first column names)

# Read in PPI and construct prior node embeddings
# Ndex2 pcnet 
if ppi == 'PCNet':
    ppi_obj = UndirectedInteractionNetwork.from_ndex(ndex_id='c3554b4e-8c81-11ed-a157-005056ae23aa', keeplargestcomponent=False,
                                                    attributes_for_names='v', node_type=int)
else:
    ppi_ = pd.read_csv(BASE_PATH+'reactome3.txt', header=0, sep='\t')
    ppi_obj = UndirectedInteractionNetwork(ppi_, keeplargestcomponent=False) #You are keeping all components, not just the largest connected one.
    
ppi_obj.set_node_types(node_types={i: "gene" for i in ppi_obj.node_names}) #set node type to gene for each node

degreedf = ppi_obj.getDegreeDF(set_index=True) #dataframe with degree of each node
degreedf.loc[['BRIP1', 'RRM2']]

# Select genes for which all features are avaialble and are situated in the network
focus_genes = sorted(list(set(ppi_obj.node_names) & set(crispr_effect.columns))) #select genes that are in both ppi and crispr
#assert len(focus_genes) == ppi_obj.N_nodes,"Error node mismatch" #12951 / 13953

focus_genes2int = {k:i for i, k in enumerate(focus_genes)}
focus_int2gene = {v:k for k, v in focus_genes2int.items()}

focus_cls = sorted(list(set(ccles.index) & set(crispr_effect.index))) #select cell lines that are in both ccles and crispr

ccles = ccles.loc[focus_cls, :] # filter only focus cell line
crispr_effect = crispr_effect.loc[focus_cls, focus_genes] # filter only focus cell line & gene
dis_groups_d = ccles.groupby('OncotreePrimaryDisease').groups
dis_groups = pd.DataFrame.from_dict(dis_groups_d, orient='index').map(lambda x: '' if x is None else x) 
dis_groups = dis_groups.apply(lambda x: ','.join(x), axis=1).apply(lambda x: [i for i in x.split(',') if i]).to_frame(name='cells')
dis_groups['length'] = dis_groups["cells"].apply(lambda x: len(x)) 
# disgroups with 2 columns: cells (list of cellines associated with disease), length (len(cells))
# OncotreePrimaryDisease is used as index

crispr_neurobl = crispr_effect.loc[dis_groups.loc[cancer_type].cells] #select rows (cell lines) from crispr_effect that are in the dis_groups for the cancer type (Neuroblastoma)
crispr_neurobl.to_csv(BASE_PATH+f"crispr_{cancer_type.replace(' ', '_')}_{ppi}.csv") #csv file contains  the filtered cell lines (columns) and genes (rows) from crispr_effect

print(crispr_neurobl.shape)

# ---------------------------------------------------------------------------------------------------------------

# define CRISPR network using SD > 0.2 + COSMIC, excluding common essentials.

common_essentials_control_df = pd.read_csv(BASE_PATH+f"Depmap/AchillesCommonEssentialControls.csv")
common_essentials_control = list([i[0].split(' ')[0] for i in common_essentials_control_df.values]) #extract the gene name

#defining non-interesting genes (e.g. Ribosomal proteins = RPL)
rpls = set([i for i in common_essentials_control if 'RPL' in i]) | set([i for i in ppi_obj.node_names if 'RPL' in i]) |\
        set([i for i in crispr_neurobl.columns if 'RPL' in i]) # remove non interesting genes
# ppi_obj.getDegreeDF(set_index=True).loc[set(rpls) & set(ppi_obj.node_names)]

std_threshold = 0.2
std_dependencies = list(crispr_neurobl.columns[crispr_neurobl.std() > std_threshold])

if remove_commonE:
    print("Removing common essentials")
    if remove_rpl: #remove common essentials and ribosomal proteins
        final_pos = list(set(std_dependencies) - set(common_essentials_control) - rpls)
    else:
        final_pos = list(set(std_dependencies) - set(common_essentials_control))
else:
    if remove_rpl: #remove ribosomal proteins
        final_pos = list(set(std_dependencies) - rpls)
    else:
        final_pos = std_dependencies

crispr_threshold_neg = -0.5
cell2dependency = {}
dependency_edgelist = []
dependency_edgelist_neg = []
X_train_dep, y_train_dep = [], []
X_test_dep, y_test_dep = [], []
for cell, row_genes in crispr_neurobl.iterrows(): #iterate over rows (cell lines) in crispr_neurobl
    tmp = row_genes[final_pos]

    tmp_pos = list(tmp[tmp < crispr_threshold_pos].index)
    tmp_neg = list(tmp[tmp > crispr_threshold_neg].index)
    ratio = len(tmp_pos)/(len(tmp_neg+tmp_pos))

    tmp_pos_train = tmp_pos[:int(len(tmp_pos)*train_ratio)]
    tmp_pos_test = tmp_pos[int(len(tmp_pos)*train_ratio):]

    tmp_neg_train = tmp_neg[:int(len(tmp_neg)*train_ratio)]
    tmp_neg_test = tmp_neg[int(len(tmp_neg)*train_ratio):]

    cell2dependency[cell] = list(tmp_pos) #store the genes that are below threshold -1,5 for each cell line
    dependency_edgelist += [[cell, i] for i in tmp_pos] 
    X_train_dep += [[cell, i] for i in tmp_pos_train+tmp_neg_train] 
    y_train_dep += [1]*len(tmp_pos_train) + [0]*len(tmp_neg_train) #label 1 for positive edges, 0 for negative edges (=dependencies)
    X_test_dep += [[cell, i] for i in tmp_pos_test+tmp_neg_test]
    y_test_dep += [1]*len(tmp_pos_test) + [0]*len(tmp_neg_test)

cells = list(crispr_neurobl.index)
disease_label = ccles.loc[cells].OncotreePrimaryDisease.values

dis2int = {e:i+1 for i, e in enumerate(set(disease_label))}
disease_label_int = [dis2int[i] for i in disease_label]

disease2color = {d: c for d, c in zip(set(disease_label), sns.color_palette("colorblind", len(set(disease_label))))}
dis_label_colors = [disease2color[i] for i in disease_label] # visualization2

# ---------------------------------------------------------------------------------------------------------------
# Combining both objects into a heterogeneous multi graph

dep_nw_obj = UndirectedInteractionNetwork(pd.DataFrame(dependency_edgelist, columns=['cell', 'gene'])) ## why GeneA GeneB column
dep_nw_obj.set_node_types(node_types={i: "cell" if i in crispr_neurobl.index else "gene" for i in dep_nw_obj.node_names})

# Create multigraph directory if it doesn't exist
multigraphs_dir = os.path.join(BASE_PATH, 'multigraphs')
os.makedirs(multigraphs_dir, exist_ok=True)

node_types = ppi_obj.node_type_names
node_types.update(dep_nw_obj.node_type_names)

mg_obj = MultiGraph({"scaffold": ppi_obj, "depmap": dep_nw_obj}, keeplargestcomponent=False, node_types=node_types)

with open(BASE_PATH+f'multigraphs/'\
          f"{cancer_type.replace(' ', '_')}_{ppi}{remove_rpl}_{useSTD}{remove_commonE}_crispr{str(crispr_threshold_pos).replace('.','_')}.pickle", 'wb') as handle:
    pickle.dump(mg_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)