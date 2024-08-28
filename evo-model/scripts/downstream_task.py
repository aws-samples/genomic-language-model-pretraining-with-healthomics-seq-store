import sys
# import math
import os
import json
from time import time
from collections import defaultdict
from pathlib import Path
from typing import List, Callable, Tuple, Dict
from contextlib import contextmanager
import logging
import argparse

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
print(f"torch {torch.__version__}")

from utilities import (timing, load_model_and_tokenizer, RMSerror,
                       WeightedAvg, Histogram)


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[Evo Training]%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)])

LOG_DIR = "/opt/ml/output/tensorboard"
tensorboard_callback = SummaryWriter(log_dir=LOG_DIR)


def parse_arguments():
    parser = argparse.ArgumentParser(
        "Training a downstream task using embeddings from an Evo model and a custom Head.")
    parser.add_argument("--model_checkpoint", type=str,
                        default='togethercomputer/evo-1-8k-base',
                        help="Model checkpoint path.")
    parser.add_argument("--model_revision", type=str,
                        default='main',
                        help="For evo model, which revision to download")
    parser.add_argument("--data_dir", type=str, default=os.environ["SM_CHANNEL_DATA"],
                        help="Path to dataset.")
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"],
                        help="Path to model output folder.")
    parser.add_argument("--epochs", type=int, required=False,
                        default=1000,
                        help="Number of training epochs.")
    parser.add_argument("--train_test_split", type=float, default=0.8,
                        help="Fraction of examples that are used for training.")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate for Adam optimizer.")
    parser.add_argument("--augment_datasets", type=int, default=0,
                        help="0 => don't augment; 1 => do augment")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="This is the batch size we use when training the head model.")
    parser.add_argument("--log_level", type=str, default="INFO",
                        help="Log Level.")
    args = parser.parse_args()
    return args


def preprocess_training_data(data_dir: str,
                             train_test_split: float,
                             augment_datasets: bool,
                             tokenizer: Callable,
                             evo_model) -> Tuple[DataLoader, DataLoader, int]:
    """
    The `data_dir` contains the file `examples.jsonl`, each line of which
    contains a DNA read (the input) and an impact score (the regression output).
    
    We use the evo-model to transform each 150-base-pair read into an embedding.
    Then we return three results:
     + a training set DataLoader that maps embeddings to impact scores;
     + similarly, an eval set DataLoader;
     + the size of the reads (they must all be the same length).
    """
    print("preprocess_training_data")
    print(f"Contents of {data_dir}:")
    for child in Path(data_dir).glob("**/*"):
        print(f" - {child}")
    with timing("Loading examples"):
        with (Path(data_dir) / "examples-processed.jsonl").open() as f:
            examples = [json.loads(line) for line in f]
    with timing("Loading embeddings"):
        with (Path(data_dir) / "examples-processed-embeddings.pt").open(mode="rb") as f:
            embeddings = torch.load(f)
    print(f"Loaded {len(examples):,} examples and {len(embeddings):,} embeddings")
    print(f"Shape of one embedding: {embeddings[0][0].shape}")
    assert len(examples) == len(embeddings)
    inputs = [ex["reads"][0] for ex in examples] # assume one read per variant for now
    targets = [ex["impact_score"] for ex in examples]
    targets = normalize_targets(targets)
    input_lens = {len(input) for input in inputs}
    assert len(input_lens) == 1, input_lens
    input_len = list(input_lens)[0]

    dataset = ListDataset(embeddings, targets)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    loader_kwargs = dict(batch_size=args.batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, **loader_kwargs)
    test_loader = DataLoader(test_dataset, **loader_kwargs)
    
    return train_loader, test_loader, input_len


def save_model(model, model_dir):
    model = model.module if hasattr(model, "module") else model
    os.makedirs(model_dir, exist_ok=True)
    checkpoint = {"state_dict": model.state_dict()}
    path = f"{model_dir}/checkpoint.pt"
    torch.save(checkpoint, path)


class CNNhead(nn.Module):
    """
    A convolutional neural net head. This "sits on top" of the
    Evo model in the sense that the inputs to this model are the
    outputs of the Evo model.
    """

    def __init__(self, dtype=torch.bfloat16, num_embedding_dims: int = 512):
        super(CNNhead, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=num_embedding_dims,out_channels=16,
                                kernel_size=3, stride=1, padding=1, dtype=dtype)
        self.max_pool_1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3,
                                  stride=1, padding=1, dtype=dtype)
        self.max_pool_2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(in_features=1_184, out_features=10, dtype=dtype)
        self.linear2 = nn.Linear(in_features=10,    out_features=10, dtype=dtype)
        self.linear3 = nn.Linear(in_features=10,    out_features=10, dtype=dtype)
        self.linear4 = nn.Linear(in_features=10,    out_features=1,  dtype=dtype)
        
    def forward(self, x):
        x = x.squeeze(1)
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = torch.relu(x) 
        x = self.max_pool_1(x)
        x = self.conv1d_2(x)
        x = torch.relu(x)
        x = self.max_pool_2(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor to [batch_size, features]
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        x = torch.sigmoid(self.linear4(x))
        return x


class ListDataset(Dataset):
    def __init__(self, inputs_list, targets_list):
        assert len(inputs_list) == len(targets_list)
        self._inputs_list = inputs_list
        self._targets_list = targets_list

    def __len__(self):
        return len(self._inputs_list)

    def __getitem__(self, idx):
        return self._inputs_list[idx], self._targets_list[idx]


def do_head_forward_pass(head: nn.Module,
                         logits: torch.tensor): # output of evo-model
    logits = logits.to("cuda")
    outputs = head(logits)
    return outputs


@contextmanager
def maybe_grad(with_grad: bool):
    if with_grad:
        yield
    else:
        with torch.no_grad():
            yield


def do_one_epoch(epoch: int,
                 head: nn.Module,
                 data_loader: DataLoader,
                 optimizer: Optimizer,
                 loss_fn,
                 training: bool):
    logger.info(f"Epoch #{epoch:,} {'training' if training else 'testing'}:")
    start_time: float = time()
    overall_loss, overall_RMS_err = WeightedAvg(), WeightedAvg()
    inputs_histogram = Histogram(n_bins=20, low=0, high=1)
    outputs_histogram = Histogram(n_bins=20, low=0, high=1)
    if training:
        head.train()
    else:
        head.eval()
    with maybe_grad(training):
        for batch_idx, (data, target) in enumerate(data_loader):
            # print(f"raw data: {data}")
            if training:
                optimizer.zero_grad()
            outputs = do_head_forward_pass(head, data[0])
            outputs = outputs.reshape((outputs.shape[0],))
            labels = target.to(dtype=torch.bfloat16, device="cuda")
            loss = loss_fn(outputs, labels)
            inputs_histogram.batch_add(labels.to("cpu").tolist())
            outputs_histogram.batch_add(outputs.to("cpu").tolist())
            if training:
                loss.backward()
                optimizer.step()
            RMS_err = RMSerror(outputs, labels)
            overall_RMS_err.add(len(data), RMS_err)
            overall_loss.add(len(data), loss.item())
    logger.info(f"Epoch #{epoch:,} "
                f"{'Train' if training else 'Eval'} Loss: {overall_loss.value()} "
                f"{'Train' if training else 'Eval'} RMSerr: {overall_RMS_err.value()} "
                f"Elapsed time: {time() - start_time:.2f} seconds")
    logger.info(f"Inputs histogram: {str(inputs_histogram)}")
    logger.info(f"Outputs histogram: {str(outputs_histogram)}")
    tensorboard_callback.add_scalar(f"Loss/{'train' if training else 'eval'}",
                                    overall_loss.value(), epoch)
    tensorboard_callback.add_scalar(f"RMSerr/{'train' if training else 'eval'}",
                                    overall_RMS_err.value(), epoch)


def normalize_targets(targets: List[float]) -> List[float]:
    """
    The loss function requires that the target be in [0..1]. This
    also matches the Sigmoid in the last layer of the Head.
    """
    min_target, max_target = min(targets), max(targets)
    return [(target - min_target)/(max_target - min_target) for target in targets]


def show_target_distribution(targets: List[float]) -> Dict[float, int]:
    target_dict = defaultdict(int)
    for target in targets:
        target_dict[target] += 1
    print(f"target distribution: {list(target_dict.items())}")
    print(f"target distribution (pct): {[(target, (count/len(targets))*100) for target, count in target_dict.items()]}")
    return target_dict


def main(args):
    model_id = args.model_checkpoint
    print(f"model_id {model_id}")
    model, tokenizer = load_model_and_tokenizer(model_id,
                                                revision=args.model_revision,
                                                download_always=True)

    train_loader, test_loader, input_len = \
        preprocess_training_data(args.data_dir, args.train_test_split,
                                 args.augment_datasets > 0,
                                 tokenizer, model)

    head = CNNhead()
    head = head.to("cuda")

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(head.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        do_one_epoch(epoch, head, train_loader, optimizer, loss_fn, training=True)
        do_one_epoch(epoch, head, test_loader,  optimizer, loss_fn, training=False)

    save_model(head, args.model_dir)


if __name__ == "__main__" :
    args = parse_arguments()
    logger.setLevel(args.log_level)
    logger.info("Starting training.......")
    main(args)