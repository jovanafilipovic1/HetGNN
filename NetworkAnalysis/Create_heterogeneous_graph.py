import os
from .MultiGraph import MultiGraph
from .UndirectedInteractionNetwork import UndirectedInteractionNetwork
import pandas as pd
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.transforms import to_undirected
from torch_geometric.data import HeteroData
from typing import Dict, List, Set, Tuple, Optional


class Create_heterogeneous_graph:
    """
    A class for creating heterogeneous graphs from protein-protein interaction and
    gene-dependency data, with added features for both gene and cell line nodes.
    
    This class handles the creation of a PyTorch Geometric HeteroData object with:
    - Two node types: genes and cell lines
    - Three edge types: gene-gene interactions, gene-cell line dependencies, and cell line-gene mutations
    - Features for both node types
    
    Attributes:
        BASE_PATH (str): Base path for data files
        cancer_type (str, optional): Type of cancer to filter cell lines by
        cell_feature (str): Type of cell line feature to use
        gene_feature (str): Type of gene feature to use
        crispr_threshold_pos (float): Threshold for positive dependency
        crispr_threshold_neg (float): Threshold for negative dependency
        std_threshold (float): Threshold for standard deviation filtering 
        metapaths (List[str], optional): List of metapaths to add to the graph
    """
    
    def __init__(self, BASE_PATH: str, cell_feature: str, gene_feature: str, cancer_type: str = None, metapaths: Optional[List[str]] = None):
        """
        Initialize the Create_heterogeneous_graph class.
        
        Args:
            BASE_PATH (str): Base path for data files
            cell_feature (str): Type of cell line feature to use (e.g., "cnv", "expression")
            gene_feature (str): Type of gene feature to use (e.g., "cgp", "bp")
            cancer_type (str, optional): Type of cancer to filter cell lines by
            metapaths (List[str], optional): List of metapaths to add to the graph.
                Valid options include: "gene_cell_gene", "cell_gene_cell", "mutation"
        """
        self.BASE_PATH = BASE_PATH
        self.cancer_type = cancer_type
        self.cell_feature = cell_feature
        self.gene_feature = gene_feature
        self.metapaths = metapaths
        self.crispr_threshold_pos = 0.8
        self.crispr_threshold_neg = 0.2
        self.std_threshold = 0.2

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, UndirectedInteractionNetwork]:
        """
        Load all the necessary datasets for building the heterogeneous graph.
        
        Returns:
            Tuple containing:
            - ccles (pd.DataFrame): Cell line metadata
            - crispr_effect (pd.DataFrame): CRISPR gene dependency scores
            - ppi_obj (UndirectedInteractionNetwork): Protein-protein interaction network
        """
        # Load cell line metadata
        ccles = pd.read_csv(self.BASE_PATH + "Depmap/Model.csv", index_col=0)
        
        # Load CRISPR gene effect data
        crispr_effect = pd.read_csv(self.BASE_PATH + 'Depmap/CRISPRGeneDependency.csv', header=0, index_col=0)
        crispr_effect.columns = [i.split(' ')[0] for i in crispr_effect.columns]  # Extract gene names
        
        # Load protein-protein interaction network
        ppi_ = pd.read_csv(self.BASE_PATH + 'reactome3.txt', header=0, sep='\t')
        ppi_obj = UndirectedInteractionNetwork(ppi_, keeplargestcomponent=False)
        ppi_obj.set_node_types(node_types={i: "gene" for i in ppi_obj.node_names})
        
        return ccles, crispr_effect, ppi_obj

    def filter_data(self, ccles: pd.DataFrame, crispr_effect: pd.DataFrame, ppi_obj: UndirectedInteractionNetwork) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Filter data to select relevant genes and cell lines. 
        For constructing the graph, genes present in the gene-cell dependency network 
        but absent from the PPI network will be filtered out.
        
        Args:
            ccles (pd.DataFrame): Cell line metadata
            crispr_effect (pd.DataFrame): CRISPR gene dependency scores
            ppi_obj (UndirectedInteractionNetwork): Protein-protein interaction network
            
        Returns:
            Tuple containing:
            - filtered_crispr (pd.DataFrame): Filtered CRISPR data
            - focus_genes (List[str]): List of relevant genes
            - focus_cls (List[str]): List of relevant cell lines
        """
        # Filter out CRISPR genes not in PPI
        ppi_genes = sorted(list(ppi_obj.node_names))
        crispr_genes_in_ppi = sorted([gene for gene in crispr_effect.columns if gene in ppi_obj.node_names])
        # Use these genes for further processing
        focus_genes = ppi_genes
        # The genes we'll use for CRISPR data filtering:
        crispr_focus_genes = crispr_genes_in_ppi
        
        # Select cell lines that are in both ccles and CRISPR
        focus_cls = sorted(list(set(ccles.index) & set(crispr_effect.index)))
        
        # Filter cell lines and genes
        filtered_ccles = ccles.loc[focus_cls, :]
        filtered_crispr = crispr_effect.loc[focus_cls, crispr_focus_genes]
        
        # Filter by cancer type if specified
        if self.cancer_type:
            cancer_cell_lines = filtered_ccles.loc[filtered_ccles["OncotreePrimaryDisease"] == self.cancer_type].index
            filtered_crispr = filtered_crispr.loc[cancer_cell_lines]
            focus_cls = sorted(list(cancer_cell_lines))
        
        return filtered_crispr, focus_genes, focus_cls

    def filter_informative_genes(self, crispr_data: pd.DataFrame) -> List[str]:
        """
        Filter for informative genes based on standard deviation and ribosomal proteins (and excluding common essential genes).
        
        Args:
            crispr_data (pd.DataFrame): CRISPR gene dependency scores
            
        Returns:
            List[str]: List of informative genes after filtering
        """
        # Load common essential genes
        common_essentials_control_df = pd.read_csv(self.BASE_PATH + "Depmap/AchillesCommonEssentialControls.csv")
        common_essentials_control = [i[0].split(' ')[0] for i in common_essentials_control_df.values]
        
        # Identify ribosomal proteins
        rpls = set([i for i in common_essentials_control if 'RPL' in i]) | \
               set([i for i in crispr_data.columns if 'RPL' in i])
        
        # Select high-variance genes (std > threshold)
        std_dependencies = list(crispr_data.columns[crispr_data.std() > self.std_threshold])
        
        # Remove common essentials and ribosomal proteins
        # informative_genes = list(set(std_dependencies) - set(common_essentials_control) - rpls)
        # Keep common essential genes, only remove ribosomal proteins and low variance genes
        informative_genes = list(set(std_dependencies) - rpls)

        return informative_genes

    def create_dependency_network(self, crispr_effect: pd.DataFrame, informative_genes: List[str]) -> pd.DataFrame:
        """
        Create a dependency network from CRISPR data using only informative genes.
        Create an edge list with labels (1 for dependency, 0 for non-dependency).
        
        Args:
            crispr_effect (pd.DataFrame): CRISPR gene dependency scores
            informative_genes (List[str]): List of informative genes to consider for dependencies
            
        Returns:
            pd.DataFrame: Dependency edge list with columns ['gene', 'cell', 'label']
        """
        dependency_edges = []
        
        for cell, row_genes in crispr_effect.iterrows():
            tmp = row_genes[informative_genes]
            
            # Add all edges with their corresponding labels
            for gene, score in tmp.items():
                if score > self.crispr_threshold_pos: # dependency
                    label = 1 
                    dependency_edges.append([gene, cell, label])
                elif score < self.crispr_threshold_neg: # non-dependency
                    label = 0   
                    dependency_edges.append([gene, cell, label])
        
        # Create dataframe with gene, cell, and label columns
        dependency_df = pd.DataFrame(dependency_edges, columns=['gene', 'cell', 'label'])
        
        return dependency_df

    def create_mutation_network(self, focus_genes: List[str], focus_cells: List[str]) -> pd.DataFrame:
        """
        Create a mutation network from somatic mutation data.
        Add edges between cell lines and genes where a mutation exists (non-zero value).
        
        Args:
            focus_genes (List[str]): List of genes to consider for mutations
            focus_cells (List[str]): List of cell lines to consider for mutations
            
        Returns:
            pd.DataFrame: Mutation edge list with columns ['cell', 'gene', 'value']
        """
        # Load mutation data
        mutation_path = self.BASE_PATH + 'Depmap/OmicsSomaticMutationsMatrixDamaging.csv'
        print(f"Loading mutation data from {mutation_path}")
        
        
        mutation_data = pd.read_csv(mutation_path, header=0, index_col=0)
        # Clean gene column names (remove any text after space if present)
        mutation_data.columns = [i.split(' ')[0] for i in mutation_data.columns]
    
        # Filter for relevant genes and cells
        relevant_genes = [g for g in focus_genes if g in mutation_data.columns]
        relevant_cells = [c for c in focus_cells if c in mutation_data.index]
        
        if len(relevant_genes) == 0 or len(relevant_cells) == 0:
            print(f"Warning: No overlapping genes or cells found in mutation data")
            print(f"Genes overlap: {len(relevant_genes)}/{len(focus_genes)}")
            print(f"Cells overlap: {len(relevant_cells)}/{len(focus_cells)}")
            return pd.DataFrame(columns=['cell', 'gene', 'value'])
        
        # Filter mutation data to relevant genes and cells
        filtered_mutation = mutation_data.loc[relevant_cells, relevant_genes]
        
        # Create edge list for non-zero values
        mutation_edges = []
        
        for cell, row in filtered_mutation.iterrows():
            for gene, value in row.items():
                if value != 0:  # Only add edges for non-zero values
                    mutation_edges.append([cell, gene, value])
        
        # Create dataframe with cell, gene, and value columns
        mutation_df = pd.DataFrame(mutation_edges, columns=['cell', 'gene', 'value'])
        
        print(f"Created mutation network with {len(mutation_df)} edges")
        return mutation_df

    def read_gmt_file(self, file_path: str, gene_list: List[str]) -> Dict[str, Set[str]]:
        """
        Read GMT file and return a dictionary of gene sets.
        
        Args:
            file_path (str): Path to the GMT file
            gene_list (List[str]): List of genes to filter by
            
        Returns:
            Dict[str, Set[str]]: Dictionary mapping gene set names to sets of genes
        """
        gene_sets = {}
        focus_genes = set(gene_list)
        
        with open(file_path) as f:
            lines = f.readlines()
            for line in lines:
                temp = line.strip('\n').split('\t')
                gene_sets[temp[0]] = set(gene for gene in temp[2:]) & focus_genes
        
        return gene_sets

    def process_gene_features(self, genes: List[str]) -> Tuple[torch.Tensor, Set[str]]:
        """
        Process gene features: identify genes with features and create feature tensor.
        
        Args:
            genes (List[str]): List of genes to process
            
        Returns:
            Tuple containing:
            - gene_feat (torch.Tensor): Gene feature tensor
            - genes_with_features (Set[str]): Set of genes with valid features
        """
        # Determine the appropriate GMT file based on gene feature type
        if self.gene_feature == 'cgp':
            gmt_file = self.BASE_PATH + "MsigDB/c2.cgp.v2023.2.Hs.symbols.gmt"
        elif self.gene_feature == 'bp':
            gmt_file = self.BASE_PATH + "MsigDB/c5.go.bp.v2023.2.Hs.symbols.gmt"
        elif self.gene_feature == 'go':
            gmt_file = self.BASE_PATH + "MsigDB/c5.go.v2023.2.Hs.symbols.gmt"
        elif self.gene_feature == 'cp':
            gmt_file = self.BASE_PATH + "MsigDB/c2.cp.v2023.2.Hs.symbols.gmt"
        elif self.gene_feature == 'C4':
            gmt_file = self.BASE_PATH + "MsigDB/c4.all.v2024.1.Hs.symbols.gmt"
        else:
            raise ValueError(f"Unsupported gene feature type: {self.gene_feature}")
        
        # Read the GMT file
        cgn = self.read_gmt_file(gmt_file, genes)
        
        # Create a feature matrix
        cgn_df = pd.DataFrame(np.zeros((len(genes), len(cgn))), index=genes, columns=list(cgn.keys()))
        for k, v in cgn.items():
            cgn_df.loc[list(v), k] = 1  # Set 1 if gene is in the gene set, 0 otherwise
            
        # Find genes with features
        genes_with_features = set(cgn_df.index[cgn_df.sum(axis=1) > 0])
        
        if len(genes_with_features) == 0:
            raise ValueError("No genes have features")
        
        # Report how many genes have no features
        genes_without_features = set(genes) - genes_with_features
        if len(genes_without_features) > 0:
            print(f"Warning: {len(genes_without_features)} out of {len(genes)} genes have no features")
        
        # If we want to create a tensor for a specific subset of genes (those with features)
        # we need to filter the dataframe first
        filtered_genes = sorted(list(genes_with_features))
        filtered_df = cgn_df.loc[filtered_genes]
        
        # Convert to tensor
        gene_feat = torch.from_numpy(filtered_df.values).to(torch.float)
        
        return gene_feat, genes_with_features

    def _create_cell_feature_tensor(self, feature_df: pd.DataFrame, valid_cells: Set[str]) -> Tuple[torch.Tensor, Dict[str, int], Set[str]]:
        """
        Helper method to create feature tensor and cell mappings.
        
        Args:
            feature_df (pd.DataFrame): DataFrame containing cell features
            valid_cells (Set[str]): Set of valid cell lines
            
        Returns:
            Tuple containing:
            - cell_feat (torch.Tensor): Cell feature tensor
            - cell2int (Dict[str, int]): Mapping from cell names to indices
            - cells_with_features (Set[str]): Final set of cells with valid features
        """
        if len(valid_cells) == 0:
            raise ValueError("No valid cells found for feature creation")
            
        # Sort cells for consistent ordering
        cells_with_features_list = sorted(list(valid_cells))
        cell2int = {c: i for i, c in enumerate(cells_with_features_list)}
        cell_feat = torch.from_numpy(feature_df.loc[cells_with_features_list].values).to(torch.float)
        
        return cell_feat, cell2int, set(cells_with_features_list)

    def _process_mosa_features(self, cells: Set[str]) -> pd.DataFrame:
        """Process MOSA latent features for cell lines."""
        # Load latent features and celline metadata
        mosa_features = pd.read_csv(self.BASE_PATH + "20250221_151226_latent_joint.csv.gz", index_col=0)
        cellines = pd.read_csv(self.BASE_PATH + "Depmap/Model.csv")
        
        # Create mapping from SangerModelID to ModelID
        sanger_to_model = dict(zip(cellines['SangerModelID'], cellines['ModelID']))
        
        # Create a new index for MOSA features using ModelID
        mosa_features['ModelID'] = mosa_features.index.map(sanger_to_model)
        mosa_features = mosa_features.dropna()  # Remove any rows where mapping failed
        
        # Set ModelID as the new index
        return mosa_features.set_index('ModelID')

    def _process_expression_features(self, cells: Set[str]) -> pd.DataFrame:
        """Process gene expression features for cell lines."""
        
        # Load expression data
        path = self.BASE_PATH + 'Depmap/OmicsExpressionProteinCodingGenesTPMLogp1.csv'
        expression = pd.read_csv(path, header=0, index_col=0)
        expression.columns = [i.split(' ')[0] for i in expression.columns]
        
        # Handle missing values with gene-specific means
        gene_means = expression.mean()
        expression = expression.fillna(gene_means)
                
        # Filter cells that are in our network
        valid_cells = cells & set(expression.index)
        if len(valid_cells) == 0:
            raise ValueError("No cells have expression data")
            
        filtered_expr = expression.loc[list(valid_cells)]
        
        if "hvg" in self.cell_feature:
            # Select high variance genes
            hvg_q = filtered_expr.std().quantile(q=0.90)
            hvg_final = filtered_expr.std()[filtered_expr.std() >= hvg_q].index

            filtered_expr = filtered_expr[hvg_final]

        if "marker_genes" in self.cell_feature:
            #load marker genes
            path_mmc5 = self.BASE_PATH+'Pacini/1-s2.0-S1535610823004440-mmc5.xls'
            df = pd.read_excel(path_mmc5, sheet_name='Extended DMAs')
            # Extract the desired columns
            marker_genes = df[['FEATURE', 'TARGET']]

            #Extract the 'FEATURES' containing '_Expr'
            expr_marker_genes = marker_genes[marker_genes['FEATURE'].str.contains('_Expr')] 
            # Extract gene names from FEATURE column by removing "_Exr" suffix
            marker_genes = expr_marker_genes['FEATURE'].str.replace('_Expr', '')
            # Find the intersection of marker genes and the columns in cancer_expression
            valid_marker_genes = marker_genes[marker_genes.isin(filtered_expr.columns)]
            unique_valid_marker_genes = valid_marker_genes.unique()
            filtered_expr = filtered_expr[unique_valid_marker_genes]

        return filtered_expr 

    def _process_cnv_features(self, cells: Set[str]) -> pd.DataFrame:
        """Process copy number variation features for cell lines."""
        path = self.BASE_PATH + 'Depmap/OmicsCNGene.csv'
        cnv = pd.read_csv(path, header=0, index_col=0)
        cnv.columns = [i.split(' ')[0] for i in cnv.columns]
        
        # Remove columns with missing values
        cnv = cnv[cnv.columns[cnv.isna().sum() == 0]]
        
        # Filter cells that are in our network
        valid_cells = cells & set(cnv.index)
        if len(valid_cells) == 0:
            raise ValueError("No cells have CNV data")
            
        filtered_cnv = cnv.loc[list(valid_cells)]
        
        # Select high variance genes
        hvg_q = filtered_cnv.std().quantile(q=0.95)
        hvg_final = filtered_cnv.std()[filtered_cnv.std() >= hvg_q].index
        return filtered_cnv[hvg_final]

    def process_cell_features(self, cells: List[str]) -> Tuple[torch.Tensor, Dict[str, int], Set[str]]:
        """
        Process cell line features based on the specified feature type.
        
        Args:
            cells (List[str]): List of cell lines to process
            
        Returns:
            Tuple containing:
            - cell_feat (torch.Tensor): Cell feature tensor
            - cell2int (Dict[str, int]): Mapping from cell names to indices
            - cells_with_features (Set[str]): Set of cells with valid features
        
        Raises:
            ValueError: If no cells have the required features or if the feature type is unsupported
        """
        cells_set = set(cells)  # Convert to set for efficient operations
        print(f"Processing features for {len(cells_set)} cells, feature type: {self.cell_feature}")
        
        # Process cell features based on specified feature type
        if self.cell_feature == "MOSA":
            features_df = self._process_mosa_features(cells_set)
        elif "expression" in self.cell_feature:
            features_df = self._process_expression_features(cells_set)
        elif self.cell_feature == "cnv":
            features_df = self._process_cnv_features(cells_set)
        else:
            raise ValueError(f"Unsupported cell feature type: {self.cell_feature}")
            
        # Find cells that have features
        valid_cells = cells_set & set(features_df.index)
        if len(valid_cells) != len(cells_set):
            missing_cells = cells_set - valid_cells
            print(f"Warning: {len(missing_cells)} cells are missing features:")
            print(sorted(list(missing_cells))[:5], "..." if len(missing_cells) > 5 else "")
            
        if not valid_cells:
            raise ValueError(f"No cells have valid features")
            
        return self._create_cell_feature_tensor(features_df, valid_cells)

    def add_metapaths(self, data: HeteroData) -> HeteroData:
        """
        Add metapaths to the HeteroData object.
        
        Supported metapaths:
        - "gene_cell_gene": Connects genes through cell dependencies
        - "cell_gene_cell": Connects cells through gene dependencies
        - "mutation": Connects cells with common mutations
        
        Args:
            data (HeteroData): The HeteroData object to add metapaths to
            
        Returns:
            HeteroData: The HeteroData object with added metapaths
        """
        if self.metapaths is None or len(self.metapaths) == 0:
            return data
            
        # Make all edge indices contiguous before adding metapaths
        for edge_type in data.edge_types:
            if hasattr(data[edge_type], 'edge_index'):
                data[edge_type].edge_index = data[edge_type].edge_index.contiguous()
        
        MP = []
        if "gene_cell_gene" in self.metapaths:
            MP.append([("gene", "dependency_of", "cell"), ("cell", "rev_dependency_of", "gene")])
        if "cell_gene_cell" in self.metapaths:
            MP.append([("cell", "rev_dependency_of", "gene"), ("gene", "dependency_of", "cell")])
        if "mutation" in self.metapaths and ('cell', 'has_mutation_in', 'gene') in data.edge_types:
            MP.append([("cell", "has_mutation_in", "gene"), ("gene", "rev_has_mutation_in", "cell")])

        print(f"Adding the following metapaths: {MP}")

        transform = T.AddMetaPaths(
            metapaths=MP,
            keep_same_node_type=True,
            drop_unconnected_node_types=False,
            weighted=True,
            drop_orig_edge_types=False,
        )
        return transform(data) 
        

    def validate_edges(self, data: HeteroData, gene2int: Dict[str, int], cell2int: Dict[str, int]) -> bool:
        """
        Validate edge indices match node mappings.
        
        Args:
            data (HeteroData): Graph data
            gene2int (Dict[str, int]): Gene to index mapping
            cell2int (Dict[str, int]): Cell to index mapping
            
        Returns:
            bool: True if all edges are valid
        """
        # Validate gene-gene edges
        gene_edges = data['gene', 'interacts_with', 'gene'].edge_index
        max_gene_idx = len(gene2int) - 1
        if torch.any(gene_edges > max_gene_idx):
            raise ValueError(f"Invalid gene index in PPI edges: max allowed {max_gene_idx}")
            
        # Validate gene-cell edges
        dep_edges = data['gene', 'dependency_of', 'cell'].edge_index
        if torch.any(dep_edges[0] > max_gene_idx):
            raise ValueError(f"Invalid gene index in dependency edges: max allowed {max_gene_idx}")
        if torch.any(dep_edges[1] > len(cell2int) - 1):
            raise ValueError(f"Invalid cell index in dependency edges: max allowed {len(cell2int) - 1}")
            
        # Validate mutation edges if they exist
        if ('cell', 'has_mutation_in', 'gene') in data.edge_types:
            mut_edges = data['cell', 'has_mutation_in', 'gene'].edge_index
            max_cell_idx = len(cell2int) - 1
            if torch.any(mut_edges[0] > max_cell_idx):
                raise ValueError(f"Invalid cell index in mutation edges: max allowed {max_cell_idx}")
            if torch.any(mut_edges[1] > max_gene_idx):
                raise ValueError(f"Invalid gene index in mutation edges: max allowed {max_gene_idx}")
            
        return True

    def run_pipeline(self) -> HeteroData:
        """
        Run the complete pipeline to create a heterogeneous graph, ensuring all nodes have features.
        
        The pipeline follows these steps:
        1. Load raw data (CRISPR, PPI, cell line metadata)
        2. Filter for genes and cells present in the networks
        3. Filter for informative genes (high std, not common essential, not RPL) to create dependency network 
        4. Process features for genes and cells
        5. Construct final heterogeneous graph with only nodes that have features
        6. Add metapaths if specified
        
        Returns:
            HeteroData: PyTorch Geometric heterogeneous graph object
        """
        print("Loading data...")
        ccles, crispr_effect, ppi_obj = self.load_data()
        
        print("Filtering data for network membership...")
        filtered_crispr, initial_focus_genes, initial_focus_cls = self.filter_data(ccles, crispr_effect, ppi_obj)

        print("Filtering for informative genes...")
        informative_genes = self.filter_informative_genes(filtered_crispr)
        
        print("Creating dependency network using informative genes...")
        dependency_df = self.create_dependency_network(filtered_crispr, informative_genes)
        
        print("Processing gene features...")
        ppi_edges = ppi_obj.getInteractionNamed() #function defined in multigraph.py
        ppi_genes = set(ppi_edges.Gene_A) | set(ppi_edges.Gene_B)
        ppi_genes = ppi_genes & set(initial_focus_genes)  # Intersect with initial focus genes
        
        dependency_genes = set(dependency_df['gene'])

        # Combine genes from both sources
        genes_in_network = sorted(list(ppi_genes | dependency_genes))
        
        # Process features for these genes
        gene_feat, genes_with_features = self.process_gene_features(genes_in_network)
        
        print("Processing cell features...")
        # Only process features for cells in the dependency network
        cells_in_network = sorted(list(set(dependency_df['cell'])))
        cell_feat, cell2int, cells_with_features = self.process_cell_features(cells_in_network)
        
        # Final set of genes and cells to include in the graph (those with features)
        final_genes = sorted(list(genes_with_features))
        final_cells = sorted(list(cells_with_features))
        
        print(f"Final network will have {len(final_genes)} genes and {len(final_cells)} cell lines")
        
        # Create gene to int mapping
        gene2int = {gene: i for i, gene in enumerate(final_genes)}
        
        # Filter PPI edges to include only genes with features
        ppi_edges = ppi_edges[
            ppi_edges.Gene_A.isin(final_genes) & 
            ppi_edges.Gene_B.isin(final_genes)
        ]
        
        # Filter dependency edges to include only genes and cells with features
        dependency_df = dependency_df[
            dependency_df['gene'].isin(final_genes) & 
            dependency_df['cell'].isin(final_cells)
        ]
        
        # Create mutation network if metapaths includes "mutation"
        mutation_df = None
        if self.metapaths and "mutation" in self.metapaths:
            print("Creating mutation network...")
            mutation_df = self.create_mutation_network(final_genes, final_cells)
            
            # Filter mutation edges to include only genes and cells with features
            if not mutation_df.empty:
                mutation_df = mutation_df[
                    mutation_df['gene'].isin(final_genes) & 
                    mutation_df['cell'].isin(final_cells)
                ]
        
        # Convert edge lists to tensors with explicit dtype and ensure contiguity
        ppi_edge_index = torch.tensor([
            [gene2int[row.Gene_A], gene2int[row.Gene_B]] 
            for _, row in ppi_edges.iterrows()
        ], dtype=torch.long).t().contiguous()
        
        # Create separate tensors for all dependency edges and their labels
        all_dep_indices = dependency_df[['gene', 'cell']].values
        all_dep_edge_index = torch.tensor([
            [gene2int[gene], cell2int[cell]] 
            for gene, cell in all_dep_indices
        ], dtype=torch.long).t().contiguous()
        
        # Create edge labels tensor (1 for dependency, 0 for non-dependency)
        edge_label = torch.tensor(dependency_df['label'].values, dtype=torch.float)
        
        # Create PyTorch Geometric HeteroData object
        data = HeteroData()
        
        # Add node IDs and names
        data['gene'].node_id = torch.tensor(list(range(len(final_genes))))
        data['gene'].names = final_genes
        data['cell'].node_id = torch.tensor(list(range(len(final_cells))))
        data['cell'].names = final_cells
        
        # Add node features - ensure they are contiguous
        data['gene'].x = gene_feat.contiguous()
        data['cell'].x = cell_feat.contiguous()
        
        # Add edge indices
        data['gene', 'interacts_with', 'gene'].edge_index = ppi_edge_index
        data['gene', 'dependency_of', 'cell'].edge_index = all_dep_edge_index
        data['gene', 'dependency_of', 'cell'].edge_label = edge_label
        
        # Add mutation edges if available
        if mutation_df is not None and not mutation_df.empty:
            mutation_edge_index = torch.tensor([
                [cell2int[cell], gene2int[gene]] 
                for cell, gene, _ in mutation_df.values
            ], dtype=torch.long).t().contiguous()
            
            # Add mutation edge values as edge attributes
            edge_attr = torch.tensor(mutation_df['value'].values, dtype=torch.float)
            
            data['cell', 'has_mutation_in', 'gene'].edge_index = mutation_edge_index
            data['cell', 'has_mutation_in', 'gene'].edge_attr = edge_attr
        
        # Validate edges before converting to undirected
        self.validate_edges(data, gene2int, cell2int)
        
        # Convert to undirected graph
        data = T.ToUndirected(merge=False)(data)
        
        # Ensure all tensors are contiguous after ToUndirected transformation
        for edge_type in data.edge_types:
            if hasattr(data[edge_type], 'edge_index'):
                data[edge_type].edge_index = data[edge_type].edge_index.contiguous()
            if hasattr(data[edge_type], 'edge_label'):
                data[edge_type].edge_label = data[edge_type].edge_label.contiguous()
        
        # Validate edges after undirected conversion
        self.validate_edges(data, gene2int, cell2int)
        
        # Validate the data
        assert data.validate(), "HeteroData validation failed"
        
        # Add metapaths if specified
        if self.metapaths:
            print("Adding metapaths to the graph...")
            data = self.add_metapaths(data)
        
        # Final validation after all transformations
        self.validate_edges(data, gene2int, cell2int)
        
        # Save the HeteroData object
        filepath = os.path.join(
            self.BASE_PATH,
            'multigraphs',
            f'heteroData_gene_cell_{self.cancer_type.replace(" ", "_") if self.cancer_type else "All"}_{self.gene_feature}_{self.cell_feature}_{"META2" if self.metapaths else ""}.pt'
        )
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(obj=data, f=filepath)
        print(f"HeteroData saved to {filepath}")
        
        return data