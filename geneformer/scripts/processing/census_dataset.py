#!urs/bin/python
"""
Script to filter, load, performf train-test split, and store CellxGene Census scRNASeq dataset as h5ad files on s3
"""
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, GroupShuffleSplit
import pandas as pd
import numpy as np
import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import subprocess

## install requirements
subprocess.check_call(
    [
       sys.executable,
       "-m",
       "pip",
       "install",
       "-r",
       f"/opt/ml/processing/input/requirements/processing_requirements.txt",
    ]
)

import cellxgene_census
#import cellxgene_census.experimental.ml as census_ml
import tiledbsoma as soma
import boto3
from sagemaker.session import Session

# boto_session = boto3.session.Session(region_name=os.environ["AWS_REGION"])
# sagemaker_session = Session(boto_session)
    

def get_obs_index_w_filter(
                           census_version="2024-07-01",
                           species="homo_sapiens",
                           value_filter="tissue=='blood' and is_primary_data==True and disease=='normal'",
                        ):
    '''
    Query cell-level meta data with filtering conditions and get obs index
    with predefined annotations(columns).
    Returns a dataframe.
    '''
    with cellxgene_census.open_soma(census_version=census_version) as census:
        obs_df = cellxgene_census.get_obs(
                    census, 
                    species, 
                    value_filter=value_filter
                )
    return obs_df


def subset_var_for_ds(
                    census_version="2024-07-01",
                    species="homo_sapiens",
                    obs_df=None,
                    presence_thresh=0.50
                   ):
    '''
    Subset and retain the transcripts that are actually measured in the selected cells/datasets provided in `obs_df`,
    set the `presence_thresh` for the minimum proportion of datasets the transcript has to be measured in.
    Returns a dataframe.
    '''
    with cellxgene_census.open_soma(census_version=census_version) as census:
        # get soma_joinid for datasets present in the obs_df
        census_datasets = (
            census["census_info"]["datasets"]
            .read(column_names=["collection_name", "dataset_title", "dataset_id", "soma_joinid"])
            .concat()
            .to_pandas()
        )
        census_datasets = census_datasets.set_index("dataset_id")
        dataset_cell_counts = pd.DataFrame(obs_df[["dataset_id"]].value_counts())
        dataset_cell_counts = dataset_cell_counts.rename(columns={0: "cell_counts"})
        dataset_cell_counts = dataset_cell_counts[dataset_cell_counts['cell_counts']!=0]
        dataset_cell_counts = dataset_cell_counts.merge(census_datasets, on="dataset_id")
        # get feature presence matrix 
        # This is a boolean matrix N x M where N is the number of datasets and M is the number of genes in the Census.
        presence_matrix = cellxgene_census.get_presence_matrix(census, "Homo sapiens", "RNA")
        # look at which transcripts were measured in the selected datasets
        presence_matrix = presence_matrix[dataset_cell_counts.soma_joinid, :]
        # use `presence_thresh` to determine which features to keep
        assert (presence_thresh > 0) and (presence_thresh <= 1)
        var_somaid = np.nonzero(presence_matrix.sum(axis=0).A1 > int(presence_matrix.shape[0] * presence_thresh))[0].tolist()
        var_df = cellxgene_census.get_var(
                    census, 
                    species, 
                    value_filter=f"soma_joinid in {var_somaid}"
        )
        return var_df

    
def train_test_split_obs(
                       obs_df,
                       train_size=0.75,
                       test_size=0.25,
                       class_key=None,
                       group_key=None,
                       shuffle=True,
                       random_state=42
            ):
    '''
    Performs train test split on cell metadata df
    '''
    # stratify by class label
    train_inds, test_inds = train_test_split(obs_df.index,
                                             test_size=test_size, 
                                             stratify=obs_df[class_key],
                                             shuffle=shuffle,
                                             random_state=random_state
                                            ) 
    # give constraint of non-overlapping groups between splits
    if group_key is not None:
        gss = GroupShuffleSplit(n_splits=1, 
                                train_size=train_size, #proportion of groups, not the proportion of samples
                                test_size=test_size, #proportion of groups, not the proportion of samples
                                random_state=random_state)
        train_index, test_index = next(gss.split(obs_df.index, obs_df[class_key], obs_df[group_key]))
        train_inds, test_inds = obs_df.index[train_index], obs_df.index[test_index]
    return train_inds.values, test_inds.values
        
    
def get_filtered_datapipe(census_version="2024-07-01",
                          species="homo_sapiens",
                          obs_query_str='',
                           var_query_str='',
                           batch_size=128,
                           shuffle=True
                ):
    '''
    Get CellxGene Census data filtered by cell-level(`obs_query_str`) and gene-level(`var_query_str`) queries, 
    and get cell type annotation.
    Returns the data in an Pytorch Dataset object
    '''
    census = cellxgene_census.open_soma(census_version=census_version)
    # get filtered data Pytorch DataSet
    experiment = census["census_data"][species]
    datapipe = census_ml.ExperimentDataPipe(
        experiment,
        measurement_name="RNA",
        X_name="raw",
        obs_query=soma.AxisQuery(value_filter=obs_query_str),
        obs_column_names=["cell_type", "dataset_id"],
        var_query=soma.AxisQuery(value_filter=var_query_str),
        batch_size=batch_size,
        shuffle=shuffle,
        soma_chunk_size=10_000,
    )
   
    census.close()
    return datapipe


def get_adata_by_index(census_version="2024-07-01",
                       species="homo_sapiens",
                       obs_coords=None,
                       var_coords=None,
                ):
    '''
    Get CellxGene Census data by desired index (`obs_coords` and `var_coords`).
    Returns the data in an AnnData object
    '''
    with cellxgene_census.open_soma(census_version=census_version) as census:
        anndata = cellxgene_census.get_anndata(
            census=census,
            organism="Homo sapiens",
            obs_coords=obs_coords,
            var_coords=var_coords,
            column_names={
                "obs": [
                    "soma_joinid",
                    "dataset_id",
                    "cell_type",
                    "tissue",
                    "assay",
                    "suspension_type",
                    "disease"
                ],
                "var": [
                    "soma_joinid",
                    "feature_id",
                    "feature_name",
                    "feature_length",
                ],
            },
        )
        return anndata

def _parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_size", type=float, default=0.75)
    parser.add_argument("--min_cellcount", type=int, default=5000)
    parser.add_argument("--local_path", type=str, default="/opt/ml/processing")
    parser.add_argument("--census_version", default="2024-07-01")
    parser.add_argument(
        "--value-filter",
        default="tissue=='blood' and is_primary_data==True and disease=='normal'",
    )
    parser.add_argument("--var_presence_thresh", type=float, default=0.8)
    parser.add_argument("-split_by_group", action='store_true')
    return parser.parse_known_args()

def main():
    ## Command line parser
    args, _ = _parse_args()
    print(args)
    # Filter datasets on the cells dimension
    obs_df = get_obs_index_w_filter(
                           census_version=args.census_version,
                           species="homo_sapiens",
                           value_filter=args.value_filter
                        )
    # Keeping cell types with more than `min_cellcount` cells
    ct_keep = obs_df['cell_type'].value_counts()[
        obs_df['cell_type'].value_counts() > args.min_cellcount
        ].index.to_list()
    obs_df = obs_df.query(f"cell_type in {ct_keep}")
    # Keeping datasets with more than 5000 cells
    ds_keep = obs_df['dataset_id'].value_counts()[
        obs_df['dataset_id'].value_counts() > args.min_cellcount
        ].index.to_list()
    obs_df = obs_df.query(f"dataset_id in {ds_keep}")
    # save final obs index
    os.makedirs(os.path.join(args.local_path, "index"), exist_ok=True)
    obs_df.to_csv(f"{args.local_path}/index/filtered_cxg_census_obs_index.csv")
    print(f"Filtering resulted in {obs_df.shape[0} cells\nSave obs index to {args.local_path}")
    # Filter dataset on the genes dimension
    var_df = subset_var_for_ds(
                    census_version=args.census_version,
                    obs_df=obs_df,
                    presence_thresh=args.var_presence_thresh
                   )
    # save final var index locally
    var_df.to_csv(f"{args.local_path}/index/filtered_cxg_census_var_index.csv")
    print(f"Save var index to {args.local_path}")
    # save the full dataset before splitting
    adata_full = get_adata_by_index(
                       obs_coords=obs_df.index.values,
                       var_coords=var_df.index.values,
    )
    adata_full.write_h5ad(os.path.join(args.local_path, "filtered_cxg_census_full.h5ad"))
    print(f"full data saved to {args.local_path}")
    # perform train test split
    if args.split_by_group:
        train_inds, test_inds = train_test_split_obs(
                           obs_df,
                           train_size=args.train_size,
                           test_size=1-args.train_size,
                           class_key='cell_type',
                           group_key='dataset_id',
                           shuffle=True,
                           random_state=42
                )
    else:
        train_inds, test_inds = train_test_split_obs(
                           obs_df,
                           train_size=args.train_size,
                           test_size=1-args.train_size,
                           class_key='cell_type',
                           shuffle=True,
                           random_state=42
                )
    print(
        f"The training data has {len(train_inds)} cells and {var_df.shape[0]} genes."
    )
    print(
        f"The test data has {len(test_inds)} cells and {var_df.shape[0]} genes."
    )
    # get train and test adata by filtering on the cell and gene dimensions and save to s3
    adata_train = get_adata_by_index(
                       obs_coords=train_inds,
                       var_coords=var_df.index.values,
    )
    adata_test = get_adata_by_index(
                       obs_coords=test_inds,
                       var_coords=var_df.index.values,
    )              
    # Save data
    for (split, adata) in zip(["train", "test"], [adata_train, adata_test]):
        os.makedirs(os.path.join(args.local_path, split), exist_ok=True)
        split_name = f"{split}_grouped_by_ds" if args.split_by_group else f"{split}_stratified_ct"
        split_output_path = os.path.join(args.local_path, f"{split}/{split_name}.h5ad")
        adata.obs.to_csv(f"{args.local_path}/index/filtered_cxg_census_obs_index_{split_name}.csv")
        print(f"{split} obs index saved to {args.local_path}/index")
        adata.write_h5ad(split_output_path)
        print(f"{split} data saved to {split_output_path}")


    
if __name__ == "__main__":
    main()
    