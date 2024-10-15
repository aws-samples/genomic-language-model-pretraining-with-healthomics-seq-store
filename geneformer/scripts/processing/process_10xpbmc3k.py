#!urs/bin/python
"""
Script to filter, load, performf train-test split, and store 10x PBMC3k scRNASeq dataset as h5ad files on s3
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import subprocess
import pickle
from collections import Counter


## install requirements
print("Installing requirements")
subprocess.check_call(
    [
       sys.executable,
       "-m",
       "pip",
       "install",
       "-r",
       "/opt/ml/processing/input/requirements/processing_requirements.txt",
    ]
)

import scanpy as sc

def train_test_split_obs(
                       obs_df,
                       test_size=0.25,
                       class_key=None,
                       group_key=None,
                       shuffle=True,
                       random_state=42
            ):
    '''
    Performs train test split on cell metadata df
    '''
    # give constraint of non-overlapping groups between splits
    if group_key is not None:
        ngroups = obs_df[group_key].nunique()
        ntest = 1 if math.floor(ngroups * test_size) == 0 else math.floor(ngroups * test_size)
        gss = GroupShuffleSplit(n_splits=1,
                                test_size=ntest, #number of groups to hold out
                                random_state=random_state)
        train_index, test_index = next(gss.split(obs_df.index, groups=obs_df[group_key]))
        train_inds, test_inds = obs_df.index[train_index], obs_df.index[test_index]
    elif class_key is not None:
        # stratify by class label
        train_inds, test_inds = train_test_split(obs_df.index,
                                             test_size=test_size, 
                                             stratify=obs_df[class_key],
                                             shuffle=shuffle,
                                             random_state=random_state
                                            )
    return train_inds.values, test_inds.values

def _parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_size", type=float, default=0.75)
    parser.add_argument("--label_colname", type=str, default="cell_type")
    parser.add_argument("--local_path", type=str, default="/opt/ml/processing")
    return parser.parse_known_args()

def main():
    ## Command line parser
    args, _ = _parse_args()
    print(args)

    data_url = "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz"
    h5ad_dir = os.path.join(args.local_path, 'h5ad_data')
    tarfile = os.path.join(h5ad_dir, "pbmc3k_filtered_gene_bc_matrices.tar.gz")
    if not os.path.exists(h5ad_dir):
        os.mkdir(h5ad_dir)
    print(f"Downloading data from {data_url}")
    subprocess.check_call(
        [
        'wget',
        "-nv",
        "-O",
        tarfile,
        data_url
        ]
    )
    subprocess.check_call(
        [
        'tar',
        "-xzf",
        tarfile,
        "-C",
        h5ad_dir
        ]
    )
    #!wget -nv -O $h5ad_dir/$dsname_filtered_gene_bc_matrices.tar.gz http://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz
    #!tar -xzf $h5ad_dir/$dsname_filtered_gene_bc_matrices.tar.gz -C $h5ad_dir/
    
    # Name of column storing cell class labels
    label_colname = args.label_colname
    
    adata = sc.read_10x_mtx(f"{h5ad_dir}/filtered_gene_bc_matrices/hg19/", var_names="gene_ids")
    adata.var["ensembl_id"] = adata.var.index
    adata.obs["n_counts"] = adata.X.sum(axis=1)
    adata.obs["joinid"] = list(range(adata.n_obs))

    # cell type annotations made by scanpy wf (https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html)
    print("preserving RAW COUNTS before counts were normalized and log transformed")
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)

    adata_hvg = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata_hvg, max_value=10)
    sc.tl.pca(adata_hvg, svd_solver="arpack")
    sc.pp.neighbors(adata_hvg, n_neighbors=10, n_pcs=40)
    sc.tl.umap(adata_hvg)

    sc.tl.leiden(adata_hvg)
    original_cell_types = [
        "CD4-positive, alpha-beta T cell (1)",
        "CD4-positive, alpha-beta T cell (2)",
        "CD14-positive, monocyte",
        "B cell (1)",
        "CD8-positive, alpha-beta T cell",
        "FCGR3A-positive, monocyte",
        "natural killer cell",
        "dendritic cell",
        "megakaryocyte",
        "B cell (2)",
    ]
    adata_hvg.rename_categories("leiden", original_cell_types)
    # annotate cell type
    adata.obs[label_colname] = adata_hvg.obs['leiden']
    print("put back raw counts to adata.X")
    adata.X = adata.layers['counts']
    # create dictionary of cell types : label ids
    target_names = list(Counter(adata.obs[label_colname]).keys())
    target_name_id_dict = dict(zip(target_names,[i for i in range(len(target_names))]))
    # save label dict
    if not os.path.exists(os.path.join(h5ad_dir, 'class_labels')):
        os.mkdir(os.path.join(h5ad_dir, 'class_labels'))
    with open(os.path.join(h5ad_dir, 'class_labels', "pbmc3k_celltype_labels.pkl"), 'wb') as handle:
        pickle.dump(target_name_id_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    adata.obs['label'] = adata.obs[label_colname].map(target_name_id_dict)

    test_size = 1 - args.train_size
    val_size = 0.05
    train_inds, test_inds = train_test_split_obs(adata.obs,
                                                test_size=test_size,
                                                class_key='label')

    train_inds, val_inds = train_test_split_obs(adata[train_inds, :].obs,
                                                test_size=val_size,
                                                class_key='label')
    for split, inds in zip(['train', 'val', 'test'], [train_inds, val_inds, test_inds]):
        save_dir = os.path.join(h5ad_dir, split)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        adata[inds, :].write(os.path.join(save_dir, f"pbmc3k_{split}.h5ad"))


if __name__ == "__main__":
    main()
