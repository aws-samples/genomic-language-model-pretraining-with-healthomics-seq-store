#!urs/bin/python
"""
Script to perform custom preprocessing on scRNASeq dataset before model inference
"""
import os
import sys
import argparse
import datetime
import json
import joblib
import time
import numpy as np
import scipy
import anndata as ad
import boto3
import shutil
from sagemaker.session import Session
from sagemaker.s3 import S3Downloader

boto_session = boto3.session.Session(region_name='us-west-2')
sagemaker_session = Session(boto_session)
print(sagemaker_session)

def download_data_from_s3(s3_uri, local_path, session):
    S3Downloader.download(s3_uri=s3_uri,
                          local_path=local_path,
                          sagemaker_session=session)

def input_fn(input_data, content_type):
    """Parse input data payload
    """
    if content_type == "text/csv":
        try:
            timestamp = datetime.datetime.now().strftime("%m%d%Y%H%M%S")
            dir_path = f'/tmp/{timestamp}'
            download_data_from_s3(input_data,
                                  local_path=dir_path,
                                  session=sagemaker_session)
            h5_path = os.path.join(dir_path, [f for f in os.listdir(dir_path) if f.endswith('.h5ad')][0])
            adata = ad.read_h5ad(h5_path)
            shutil.rmtree(dir_path)
            print(f"{dir_path} removed successfully!")
        except Exception as inst:
            print(type(inst))    # the exception type
            print(inst.args)     # arguments stored in .args
            print(inst)     
            print('Failed to access data')
    else:
        raise ValueError("{} not supported by script!".format(content_type))
    return adata


def predict_fn(adata, model):
    """Preprocess input data and get prediction from model
    Transform count matrix: subsetting selected features, normalize and log-transform counts for prediction
    """
    mdl, feature_list = model
    # for now only support when all the feature ensembl ids are detected in query data
    assert "ensembl_id" in adata.var, "Missing ensembl_id in adata.var"
    assert np.all([f in adata.var["ensembl_id"] for f in feature_list]), \
        f"These gene features in the model are missing from the query data: {[ f for f in feature_list if (f not in adata.var['ensembl_id']) ]}"
    adata = adata[:, feature_list]
    # if contains raw counts, normalize and log-transform raw counts
    if adata.raw:
        X = adata.raw.X.toarray()
        X = np.log1p(X * 1e4 / X.sum(axis=1)[:, np.newaxis])
    else:
        X = adata.X
        if scipy.sparse.issparse(X):
            X = X.toarray()
        # if X is all integers, normalize and log-transform X
        if np.all(np.equal(np.mod(X, 1), 0)):
            X = np.log1p(X * 1e4 / X.sum(axis=1)[:, np.newaxis])
    return mdl.predict(X)


def model_fn(model_dir):
    """Deserialize fitted model"""
    mdl = joblib.load(os.path.join(model_dir, "model.joblib"))
    feature_list = list(joblib.load(os.path.join(model_dir, "feature_names.joblib")))
    return mdl, feature_list

