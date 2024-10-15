#!urs/bin/python
"""
Script to train and eval baseline logistic regression model on 10x PBMC3k scRNASeq dataset
"""
import os
import sys
import argparse
import datetime
import pickle
import json
import joblib
import time
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay, balanced_accuracy_score
import subprocess
import boto3
import mlflow
import matplotlib.pyplot as plt
import scanpy as sc
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def compute_metrics(mdl, X, y):
    '''
    Function to evaluate model on specified metrics for transformer Trainer
    '''
    start = time.time()
    y_pred = mdl.predict(X)
    end = time.time()
    test_runtime = (end - start)
    test_samples_per_second = X.shape[0] / test_runtime
    y_true = y
    acc = accuracy_score(y_true, y_pred)
    class_ave_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    global_f1 = f1_score(y_true, y_pred, average='micro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')

    return {
        'test_runtime': test_runtime,
      'test_samples_per_second': test_samples_per_second,
      'test_accuracy': acc,
      "test_class_averaged_accuracy": class_ave_acc,
      'test_macro_f1': macro_f1,
      'test_global_f1': global_f1,
      'test_class_weighted_f1': weighted_f1
    }

def plot_confusion_matrix(
    y_true, y_pred, classes, title, save_dir
    ):
    '''
    Plot classification confusion matrix
    '''
    fig = plt.figure()
    fig.set_size_inches(10, 10)
    display = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=classes, sample_weight=None) #, normalize='true'
    plt.xticks(rotation=90)
    plt.title(title)
    plt.savefig(os.path.join(save_dir, f'{title}.png'), dpi=fig.dpi)

    
def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_weight", type=str, default="balanced")
    parser.add_argument("--penalty", type=str, default='l2')
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--solver", type=str, default="lbfgs")
    parser.add_argument("--label_colname", type=str, default="label")
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--output_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--class_label", type=str, default=os.environ["SM_CHANNEL_LABELS"])
    return parser.parse_known_args()

def main():
    # Parse arguments
    args, _ = _parse_args()
    print("args:", args)
    # set up mlflow
    if 'MLFLOW_TRACKING_ARN' in os.environ:
        # Set the Tracking Server URI using the ARN of the Tracking Server you created
        mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_ARN'])
        # Enable autologging in MLflow
        mlflow.autolog()

    # specify hyperparameters
    penalty = args.penalty
    class_weight = args.class_weight
    max_iter = args.max_iter
    solver = args.solver

    with open(os.path.join(args.class_label, "pbmc3k_celltype_labels.pkl"), 'rb') as handle:
        target_name_id_dict = pickle.load(handle)
    logger.info('target_name_id_dict: %s',target_name_id_dict)

    logger.info(f"train dir: {args.train}")
    logger.info(f"test dir: {args.test}")
    adata_train = sc.read_h5ad(os.path.join(args.train, "pbmc3k_train.h5ad"))
    # keep normalized counts of highly variable genes as features
    adata_train = adata_train[:, adata_train.var.highly_variable]

    y_train = adata_train.obs[args.label_colname]
    X_train = adata_train.X.toarray()
    
    # save selected feature names used by the model
    feature_list = adata_train.var.loc[adata_train.var.highly_variable, "ensembl_id"].tolist()
    joblib.dump(feature_list, os.path.join(args.model_dir, "feature_names.joblib"))
    clf = LogisticRegression(random_state=0,
                            multi_class="ovr",
                            penalty=penalty,
                            class_weight=class_weight,
                            max_iter=max_iter,
                            solver=solver,
                            n_jobs=-1)
    start = time.time()
    clf = clf.fit(X_train, y_train)
    end = time.time()
    train_time_per_sample = (end - start) / X_train.shape[0]
    logger.info(f"train_time_per_sample= {train_time_per_sample}")

    # save the model
    current_date = datetime.datetime.now()
    datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}_{current_date.strftime('%X').replace(':','')}"

    joblib.dump(clf, os.path.join(args.model_dir, f"{datestamp}_logistic_Classifier_penalty{penalty}_clsweight{class_weight}_maxiter{max_iter}_solver{solver}.joblib"))
    logger.info("saved model!")

    # eval on train and test set
    adata_test = sc.read_h5ad(os.path.join(args.test, "pbmc3k_test.h5ad"))
    # keep normalized counts of highly variable genes as features
    adata_test = adata_test[:, adata_test.var.highly_variable]
    y_test = adata_test.obs[args.label_colname]
    X_test = adata_test.X.toarray()

    output_dir = args.output_dir
    class_names = [*target_name_id_dict.keys()]
    
    for split, X_eval, y_eval in zip(['train', 'test'],
                                     [X_train, X_test],
                                     [y_train, y_test]):
        metrics = compute_metrics(clf, X_eval, y_eval)
        with open(os.path.join(f"{output_dir}", f"{split}_metrics.json"), "w") as fp:
            json.dump(metrics, fp)
        logger.info(f"metrics saved to {output_dir}")
        plot_confusion_matrix(y_eval,
                          clf.predict(X_eval),
                          classes=class_names,
                          title=f'Logistic regression classifier prediction on {split} set',
                         save_dir=output_dir)


if __name__ == "__main__":
    main()
