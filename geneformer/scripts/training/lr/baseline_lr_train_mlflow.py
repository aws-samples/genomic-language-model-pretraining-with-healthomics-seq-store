#!urs/bin/python
"""
Script to train and eval baseline logistic regression model on a scRNASeq dataset
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
# set up ml flow params
parent_run_id = os.environ.get('MLFLOW_PARENT_RUN_ID', None)
mlflow_experiment_name = os.environ.get('MLFLOW_EXPERIMENT_NAME', None)
logger.info(mlflow_experiment_name)

def compute_metrics(mdl, X, y, dataset_type="eval"):
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
        f'{dataset_type}_runtime': test_runtime,
      f'{dataset_type}_samples_per_second': test_samples_per_second,
      f'{dataset_type}_accuracy': acc,
      f'{dataset_type}_class_averaged_accuracy': class_ave_acc,
      f'{dataset_type}_macro_f1': macro_f1,
      f'{dataset_type}_global_f1': global_f1,
      f'{dataset_type}_class_weighted_f1': weighted_f1
    }

def plot_confusion_matrix(
    y_true, y_pred, classes, title, save_dir
    ):
    '''
    Plot classification confusion matrix
    '''
    fig = plt.figure()
    display = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=classes, sample_weight=None) #, normalize='true'
    plt.xticks(rotation=90)
    plt.title(title)
    plt.savefig(os.path.join(save_dir, f'{title}.png'), dpi=fig.dpi)
    return fig


def train(args):
    '''Model train function'''
    # specify hyperparameters
    penalty = args.penalty
    class_weight = args.class_weight
    max_iter = args.max_iter
    solver = args.solver
    dataset_name = args.dataset_name
    # if there's a parent_run_id run as nested MLflow_run
    nested = False
    if parent_run_id:
        nested = True

    logger.info(f"train dir: {args.train}")
    logger.info(f"test dir: {args.test}")
    adata_train = sc.read_h5ad(os.path.join(args.train, f"{dataset_name}_train.h5ad"))
    # keep normalized counts of highly variable genes as features
    adata_train = adata_train[:, adata_train.var.highly_variable]

    y_train = adata_train.obs[args.label_colname]
    X_train = adata_train.X.toarray()
    logger.info(f"training data shape: {X_train.shape}")

    with mlflow.start_run(nested=nested) as run:
        params = {
            k: o
            for k, o in vars(args).items()
        }
        sm_training_env = json.loads(os.environ['SM_TRAINING_ENV'])
        job_name = sm_training_env['job_name']
        region = os.getenv('AWS_REGION')
        job_uri = f'https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/jobs/{job_name}'
        mlflow.log_params(
            {**params, 'sagemaker_job_name': job_name, 'sagemaker_job_uri': job_uri}
        )
        # save selected feature names used by the model
        feature_list = adata_train.var.loc[adata_train.var.highly_variable, "ensembl_id"].tolist()
        joblib.dump(feature_list, os.path.join(args.model_dir, "feature_names.joblib"))
        mlflow.log_artifact(os.path.join(args.model_dir, "feature_names.joblib"))
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
        mlflow.log_metric("train_time_per_sample", train_time_per_sample)

        # save the model
        current_date = datetime.datetime.now()
        datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}_{current_date.strftime('%X').replace(':','')}"

        joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))
        #{datestamp}_logistic_Classifier_penalty{penalty}_clsweight{class_weight}_maxiter{max_iter}_solver{solver}.joblib"))
        logger.info("saved model!")
        
        #report_metrics(run, clf, args, args.train, dataset_name, dataset_type="train")
        report_metrics(run, clf, args, args.test, dataset_name, dataset_type="test")

def report_metrics(run, model, args, data_path, dataset_name, dataset_type="test"):
    """evaluate model and log metrics"""
    adata_eval = sc.read_h5ad(os.path.join(data_path, f"{dataset_name}_{dataset_type}.h5ad"))
    # keep normalized counts of highly variable genes as features
    adata_eval = adata_eval[:, adata_eval.var.highly_variable]
    y_eval = adata_eval.obs[args.label_colname]
    X_eval = adata_eval.X.toarray()

    output_dir = args.output_dir
    with open(os.path.join(args.class_label, f"{dataset_name}_celltype_labels.pkl"), 'rb') as handle:
        target_name_id_dict = pickle.load(handle)
    logger.info('target_name_id_dict: %s',target_name_id_dict)
    class_names = [*target_name_id_dict.keys()]

    metrics = compute_metrics(model, X_eval, y_eval)
    logger.info(f"metrics: %s", metrics)
    mlflow.log_metrics(metrics)
    predictions = model.predict(X_eval)
    fig = plot_confusion_matrix(y_eval,
                      predictions,
                      classes=class_names,
                      title=f'Logistic regression classifier prediction on {dataset_type} set',
                     save_dir=output_dir)
    mlflow.log_figure(fig, f'Logistic regression classifier prediction on {dataset_type} set.png')
    mlflow.log_artifact(output_dir)

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_weight", type=str, default="balanced")
    parser.add_argument("--penalty", type=str, default='l2')
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--solver", type=str, default="lbfgs")
    parser.add_argument("--label_colname", type=str, default="label")
    parser.add_argument("--dataset_name", type=str, default="pbmc3k")
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
    #logger.info("args:", args)

    mlflow.set_experiment(mlflow_experiment_name)
    if parent_run_id:
        with mlflow.start_run(run_id=parent_run_id):
            train(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
