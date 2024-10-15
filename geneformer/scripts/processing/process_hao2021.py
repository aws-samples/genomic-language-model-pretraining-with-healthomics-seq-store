#!urs/bin/python
"""
Script to download, calculate HVG, performf train-test split, and store 
Hao2021 Cell (https://doi.org/10.1016/j.cell.2021.04.048) scRNASeq dataset as h5ad files on s3
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
import math

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
    parser.add_argument("-split_by_group", action='store_true')
    parser.add_argument("--group_key", type=str, default="orig.ident")
    return parser.parse_known_args()


def main():
    ## Command line parser
    args, _ = _parse_args()
    print(args)
    
    # download Hao et al. 2021 Cell dataset from cellxgene
    data_url = "https://datasets.cellxgene.cziscience.com/55c120dc-6a20-4caf-9513-f5970b24b1be.h5ad"
    h5ad_dir = os.path.join(args.local_path, 'h5ad_data')
    if not os.path.exists(h5ad_dir):
        os.mkdir(h5ad_dir)
    print(f"Downloading data from {data_url}")
    subprocess.check_call(
        [
        'wget',
        "-nv",
        "-O",
        os.path.join(h5ad_dir, "hao2021.h5ad"),
        data_url
        ]
    )

    # Name of column storing cell class labels
    label_colname = args.label_colname

    adata = sc.read_h5ad(os.path.join(h5ad_dir, "hao2021.h5ad"))
    # get rid of some unused attributes
    del adata.obsm
    del adata.uns
    del adata.obsp
    del adata.layers
    del adata.varm
    adata.var["ensembl_id"] = adata.var.index
    adata.obs["n_counts"] = adata.obs["nCount_RNA"]

    # get the highly variable genes
    print("use RAW COUNTS to calculate hvg")
    adata.X = adata.raw.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata)

    print("put back raw counts to adata.X")
    adata.X = adata.raw.X.copy()
    del adata.raw
    # create dictionary of cell types : label ids
    target_names = list(Counter(adata.obs[label_colname]).keys())
    target_name_id_dict = dict(zip(target_names,[i for i in range(len(target_names))]))
    # save label dict
    if not os.path.exists(os.path.join(h5ad_dir, 'class_labels')):
        os.mkdir(os.path.join(h5ad_dir, 'class_labels'))
    with open(os.path.join(h5ad_dir, 'class_labels', "hao2021_pbmc_celltype_labels.pkl"), 'wb') as handle:
        pickle.dump(target_name_id_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    adata.obs['label'] = adata.obs[label_colname].map(target_name_id_dict)
    test_size = 1 - args.train_size
    val_size = 0.05
    if args.split_by_group and (args.group_key in adata.obs):
        train_inds, test_inds = train_test_split_obs(adata.obs,
                                                test_size=test_size,
                                                group_key=args.group_key)

        train_inds, val_inds = train_test_split_obs(adata[train_inds, :].obs,
                                                test_size=val_size,
                                                class_key='label')
    else:
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
        adata[inds, :].write(os.path.join(save_dir, f"hao2021_pbmc_{split}.h5ad"))


if __name__ == "__main__":
    main()
