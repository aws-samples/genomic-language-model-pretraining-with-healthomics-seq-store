
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, set_seed
from peft import get_peft_model, LoraConfig, TaskType

from datasets import Dataset
from datasets import load_dataset
import evaluate

import torch
import torch.distributed as dist
import numpy as np
from omics_utils import load_dataset_from_omics

import os
import logging
import argparse
import sys


# Set up logging
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.getLevelName("INFO"),
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Set tensorboard 
LOG_DIR = "/opt/ml/output/tensorboard"


def as_boolean(input_str: str) -> bool:
    return input_str.lower() == "true"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default="us-east-1")  # needed for boto3 client(s) - omics

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument(
        "--bf16",
        type=bool,
        default=True if torch.cuda.get_device_capability()[0] == 8 else False,
        help="Whether to use bf16.",
    )
    # caduceus model
    parser.add_argument("--model_name", type=str, default="kuleshov-group/caduceus-ps_seqlen-1k_d_model-118_n_layer-4_lr-8e-3")
    parser.add_argument("--sequence_store_id", type=str)
    parser.add_argument("--learning_rate", type=str, default=1e-3)
    parser.add_argument("--weight_decay", type=str, default=0.01)
    parser.add_argument("--benchmark_name", type=str, default="demo_human_or_worm")
    parser.add_argument("--peft", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=42)

    # Data, model, and output directories
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test_dir", type=str, default=os.environ.get("SM_CHANNEL_TEST"))

    args, _ = parser.parse_known_args()
    return args


def train(args):
    set_seed(args.seed)
    
    # download model from model hub
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        pad_token_id=tokenizer.pad_token_id,
        num_labels=3 if args.benchmark_name == "human_ensembl_regulatory" else 2,
    )
    # CRITICAL STEP: initialize weights of output layer for better stability
    model.score.weight.data.normal_(std=0.02)
    
    if as_boolean(str(args.peft)):
        logger.info("Using PEFT for fine-tuning")
        # todo: add these params to parser
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, 
            inference_mode=False, 
            r=8, 
            lora_alpha=32, 
            lora_dropout=0.1,
            target_modules=["in_proj", "x_proj", "dt_proj", "out_proj"],
            modules_to_save=["score"],
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Num. trainable params: {num_params}")

    metric = evaluate.load("accuracy")

    # compute metrics function for binary classification
    def compute_metrics(eval_pred):
        """Computes accuracy on a batch of predictions"""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)

    def tokenize_function(examples):
        return tokenizer(examples["seq"], padding="max_length", truncation=True)

    def labels_to_list(examples):
        return {"labels": torch.tensor(float(examples["label"]), dtype=torch.float)}
    
    def prepare_tokenized_dataset(dataset: Dataset):
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset.map(labels_to_list, batched=False)

    # load datasets
    if args.sequence_store_id:
        logger.info(f"Loading Omics Datasets from sequence store id: {args.sequence_store_id}")
        train_and_eval_dataset_raw = load_dataset_from_omics(
            sequence_store_id=args.sequence_store_id,
            task=args.benchmark_name,
            split="train",
            region_name=args.region,
        )
        test_dataset_raw = load_dataset_from_omics(
            sequence_store_id=args.sequence_store_id,
            task=args.benchmark_name,
            split="test",
            region_name=args.region,
        )
    else:
        logger.info("Loading datasets from HuggingFace")
        dataset_name = f"katarinagresova/Genomic_Benchmarks_{args.benchmark_name}"
        train_and_eval_dataset_raw = load_dataset(dataset_name, split="train")
        test_dataset_raw = load_dataset(dataset_name, split="test")
    
    # tokenize and further split train into train/validation
    train_and_eval_dataset = prepare_tokenized_dataset(train_and_eval_dataset_raw).train_test_split(test_size=0.1, seed=42)

    train_dataset, eval_dataset = train_and_eval_dataset["train"], train_and_eval_dataset["test"]
    test_dataset = prepare_tokenized_dataset(test_dataset_raw)

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(test_dataset)}")

    logger.info(f"Learning Rate: {float(args.learning_rate)}")
    logger.info(f"Weight Decay: {float(args.weight_decay)}")

    # define training args
    training_args = TrainingArguments(
        output_dir="/tmp",
        overwrite_output_dir=True,
        learning_rate=float(args.learning_rate),
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=float(args.weight_decay),
        warmup_steps=args.warmup_steps,
        logging_dir=LOG_DIR,
        eval_strategy="epoch",  # in transformers >= 4.38 it is `eval_strategy`
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        save_safetensors=False,
        load_best_model_at_end=True,
        bf16=args.bf16,
    )

    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,    # in transformers >= 4.38 it is `processing_class`
        compute_metrics=compute_metrics,
    )

    # train model
    trainer.train()
    
    # evaluate model on reserved test data
    logger.info("Evaluating on test set")
    eval_result = trainer.evaluate(eval_dataset=test_dataset)

    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
        logger.info("***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            logger.info(f"{key} = {value}\n")
            writer.write(f"{key} = {value}\n")

    # Saves the model to s3
    trainer.save_model(args.model_dir)


def main():
    args = parse_args()
    logger.info(f"Got args: {args}")
    train(args)


if __name__ == "__main__":
    main()