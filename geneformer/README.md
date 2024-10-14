# Fine Tune and Benchmark Geneformer (Single cell RNA-Seq foundation model) For Cell Type/Cell State Classification

This project is a POC for apply machine learning in Single Cell Transcriptomics (scRNA-Seq) to perform cell type annotation.

scRNA-Seq has been gaining popularity in drug discovery research and therapeutic development in recent years. scRNA-Seq provides a high resolution view of gene expression at a single cell level, revealing the heterogeneity of cell populations in healthy and disease states.

## Architecture overview

![POC architecture overview](./images/architecture.png)

This architecture consists of two parts.

- The first [notebook](./AHO_r2r_StarSolo.ipynb) demonstrates getting from raw NGS files to gene expression count matrices using AWS HealthOmics Ready2Run workflow.
- The second [notebook](./Finetune_Geneformer_SM_Jobs_MLFlow.ipynb) demonstrates using AWS SageMaker to perform end-to-end machine learning workflow, including data preprocessing, training a logistic regression baseline model, fine tuning Geneformer for cell type classification task, hyperparameter tuning, MLflow experiment tracking, and model deployment.

## Background

### Machine learning application to automate cell type annotation
Traditional methods for cell type annotation using scRNA-Seq are manual, usually dependent on pre-defined lists of cell type-specific genes, and difficult to standardize and scale to multiple datasets. 

![Automating](./images/celltype%20annotation.png)
Clarke et al. 2021 https://www.nature.com/articles/s41596-021-00534-0

### Geneformer - an scRNA-Seq foundation model 
Geneformer is a BERT-like transformer model pre-trained on data from about 95M single-cell transcriptomes across various human tissues. During the pretraining phase, the model employs a masked language modeling technique. 
This project aims to demonstrate the process of fine-tuning this foundation model to perform cell type annotation, and compares the performance of this approach to a linear regression classification model trained from scratch.

![Geneformer](./images/geneformer.png)
Theodoris et al. 2023 https://doi.org/10.1038/s41586-023-06139-9
