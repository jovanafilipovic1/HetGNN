# MOSA - Multi-omic Synthetic Augmentation

This repository presents a bespoke Variational Autoencoder (VAE) that integrates all molecular and phenotypic data sets available for cancer cell lines. This model was used to create cell line embeddings as input for the graph neural network.

![MOSA Overview](./figure/MOSA_Overview.png)

## Instructions
The original code can be found on `https://github.com/QuantitativeBiology/PhenPred`. 

I used MOSA with custom data: the paths to the data files are configured in `reports/vae/files/Jovana_hyperparameters.json` and the data can be found in the figshare repositories (`https://doi.org/10.6084/m9.figshare.24420580` and `https://doi.org/10.6084/m9.figshare.24420598`).

jobscript_VAE.sh was used to run the script on the HPC.

The embeddings can be found in `./reports/vae/files`.

## Citation
Cai, Z et al., Synthetic multi-omics augmentation of cancer cell lines using unsupervised deep learning, 2023
