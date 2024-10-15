#!urs/bin/python
"""
Script to fine tune Geneformer on 10x PBMC3k scRNASeq dataset
"""
import os
import sys
import argparse
import datetime
import datasets
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay, balanced_accuracy_score
from transformers import BertForSequenceClassification
from transformers import Trainer
from transformers.training_args import TrainingArguments
from geneformer import TranscriptomeTokenizer
from geneformer import DataCollatorForCellClassification, TOKEN_DICTIONARY_FILE
import pandas as pd
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import time
import logging
import subprocess
## install mlflow dependencies
print("Installing mlflow dependencies")
subprocess.check_call(
    [
       sys.executable,
       "-m",
       "pip",
       "install",
       "mlflow==2.13.2",
       "sagemaker-mlflow==0.1.0"
    ]
)

import mlflow

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
# set up ml flow params
parent_run_id = os.environ.get('MLFLOW_PARENT_RUN_ID', None)
mlflow_experiment_name = os.environ.get('MLFLOW_EXPERIMENT_NAME', None)
logger.info(mlflow_experiment_name)


# def compute_metric_with_extra(datatype):
#     '''Function factory to pass extra data to compute_metrics'''
#     def compute_metrics(preds):
#         '''
#         Function to evaluate model on specified metrics for transformer Trainer
#         '''
#         labels = pred.label_ids
#         preds = pred.predictions.argmax(-1)
#         acc = accuracy_score(labels, preds)
#         class_ave_acc = balanced_accuracy_score(labels, preds)
#         macro_f1 = f1_score(labels, preds, average='macro')
#         global_f1 = f1_score(labels, preds, average='micro')
#         weighted_f1 = f1_score(labels, preds, average='weighted')
#         return {
#           f'{datatype}_accuracy': acc,
#           f'{datatype}_class_averaged_accuracy': class_ave_acc,
#           f'{datatype}_macro_f1': macro_f1,
#           f'{datatype}_global_f1': global_f1,
#           f'{datatype}_class_weighted_f1': weighted_f1
#         }
#     return compute_metrics

# compute_metric = compute_metric_with_extra(extra_data)

def compute_metrics(pred):
    '''
    Function to evaluate model on specified metrics for transformer Trainer
    '''
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    class_ave_acc = balanced_accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    global_f1 = f1_score(labels, preds, average='micro')
    weighted_f1 = f1_score(labels, preds, average='weighted')
    return {
      'accuracy': acc,
      "class_averaged_accuracy": class_ave_acc,
      'macro_f1': macro_f1,
      'global_f1': global_f1,
      'class_weighted_f1': weighted_f1
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


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--num_proc", type=int, default=16)
    parser.add_argument("--model_name", type=str, default="gf-6L-30M-i2048")
    parser.add_argument("--max_lr", type=float, default=5e-5)
    parser.add_argument("--freeze_layers", type=int, default=0)
    parser.add_argument("--geneformer_batch_size", type=int, default=28)
    parser.add_argument("--lr_schedule_fn", type=str, default="linear")
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--label_colname", type=str, default="label")
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--class_label", type=str, default=os.environ["SM_CHANNEL_LABELS"])
    return parser.parse_known_args()

def main():
    ## Command line parser
    args, _ = _parse_args()
    print(args)
    # set model parameters
    model_name = args.model_name
    # max input size
    max_input_size = int(model_name.split('-i')[-1]) #2 ** 11  # 2048
    print(max_input_size)
    # special token
    mdl_special_token = True if '95M' in model_name else False
    print(mdl_special_token)
    # set training parameters
    # max learning rate
    max_lr = args.max_lr
    # how many pretrained layers to freeze
    freeze_layers = args.freeze_layers
    # number gpus
    num_gpus = args.num_gpus
    # number cpu cores
    num_proc = args.num_proc
    # batch size for training and eval. Note that during train cycle space will look free but eval will fill it
    geneformer_batch_size = args.geneformer_batch_size
    # learning schedule
    lr_schedule_fn = args.lr_schedule_fn
    # warmup steps
    warmup_steps = args.warmup_steps
    # number of epochs
    epochs = args.epochs
    # optimizer
    optimizer = args.optimizer

    # Name of column storing cell class labels
    label_colname = args.label_colname

    tokenized_input_dir = os.path.join(args.output_data_dir, "tokenized_data")
    tokenized_input_prefix = f"{model_name}"

    if not os.path.exists(tokenized_input_dir):
        os.makedirs(tokenized_input_dir)

    #For 95M model series, special_token should be True and model_input_size should be 4096. 
    # For 30M model series, special_token should be False and model_input_size should be 2048.

    tokenizer = TranscriptomeTokenizer(custom_attr_name_dict={
                                                            "joinid": "joinid",
                                                            #'cell_type': "cell_type",
                                                            label_colname: label_colname
                                                            #"donor_id": "donor_id",
                                                            #"disease": "disease"
                                                            },
                                        model_input_size=max_input_size,
                                        #special_token=mdl_special_token,
                                        nproc=num_proc)
    for split, data_dir in zip(['train', 'test'], [args.train, args.test]):
        tokenizer.tokenize_data(
            data_directory=data_dir,
            output_directory=tokenized_input_dir,
            output_prefix=tokenized_input_prefix+f"_{split}",
            file_format="h5ad",
        )

    with open(os.path.join(args.class_label, "pbmc3k_celltype_labels.pkl"), 'rb') as handle:
        target_name_id_dict = pickle.load(handle)
    print(target_name_id_dict)

    train_dataset = load_from_disk(os.path.join(tokenized_input_dir, tokenized_input_prefix + "_train.dataset"))
    test_dataset = load_from_disk(os.path.join(tokenized_input_dir, tokenized_input_prefix + "_test.dataset"))

    # set logging steps
    logging_steps = round(len(train_dataset)/geneformer_batch_size/10)
    # geneformer repo download path inside the container
    pretrained_model_path = os.path.join("/Geneformer", model_name)

    model = BertForSequenceClassification.from_pretrained(pretrained_model_path,
                                                  num_labels=len(target_name_id_dict),
                                                  output_attentions = False,
                                                  output_hidden_states = False)
    if num_gpus == 1:
        model.to("cuda")
    elif num_gpus == 0:
        model.to("cpu")
        print("Using CPU for training")
    else:
        model.to("cuda:0")
        print("Multiple GPUs not supported. Only using one GPU at this point")

    # freeze the specified number of layers on the encoder
    if freeze_layers > 0:
        modules_to_freeze = model.bert.encoder.layer[:freeze_layers]
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False

    training_args = {
        "learning_rate": max_lr,
        "do_train": True,
        "do_eval": True,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_steps": logging_steps,
        "group_by_length": True,
        "length_column_name": "length",
        "disable_tqdm": False,
        "lr_scheduler_type": lr_schedule_fn,
        "warmup_steps": warmup_steps,
        "weight_decay": 0.001,
        "per_device_train_batch_size": geneformer_batch_size,
        "per_device_eval_batch_size": geneformer_batch_size,
        "num_train_epochs": epochs,
        "load_best_model_at_end": True,
        "output_dir": args.output_data_dir,
        "eval_accumulation_steps": 1,  # avoid running out of memory during eval
        "fp16": False,
        #"resume_from_checkpoint":"checkpoint"
    }

    training_args_init = TrainingArguments(**training_args)

    with open(TOKEN_DICTIONARY_FILE, "rb") as f:
        token_dict = pickle.load(f)

    trainer = Trainer(
        model=model,
        args=training_args_init,
        data_collator=DataCollatorForCellClassification(token_dictionary=token_dict),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    nested = False
    if parent_run_id:
        nested = True
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
        start = time.time()
        trainer.train()
        end = time.time()
        train_time_per_sample = (end - start) / epochs / len(train_dataset)
        logger.info(f"train_time_per_sample= {train_time_per_sample}")
        mlflow.log_metric("train_time_per_sample", train_time_per_sample)
        trainer.save_model(args.model_dir)
        # evaluate model
        # eval_result = trainer.evaluate(eval_dataset=test_dataset)

        # eval post fine tuning
        for split, ds in zip(['test'], [test_dataset]):
            predictions_ft = trainer.predict(ds)
            with open(os.path.join(f"{args.output_data_dir}", f"{split}set_predictions.pickle"), "wb") as fp:
                pickle.dump(predictions_ft, fp)
            trainer.save_metrics(f"{split}set_metrics", predictions_ft.metrics)
            y_pred = np.argmax(predictions_ft.predictions, axis=1)
            y_true = ds['label']
            class_names = [*target_name_id_dict.keys()]
            fig = plot_confusion_matrix(y_true, y_pred,
                                  classes=class_names,
                                  title=f'Finetuned mdl prediction on {split} set',
                                  save_dir=args.output_data_dir)
            #mlflow.log_figure(fig, f'Finetuned mdl prediction on {split} set.png')
        mlflow.log_artifact(args.output_data_dir)

        
if __name__ == "__main__":
    main()
