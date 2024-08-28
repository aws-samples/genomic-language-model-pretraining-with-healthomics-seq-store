import sys
import os
import json
from random import choice
from collections import defaultdict
from pathlib import Path
from typing import List, Callable, Tuple, Dict
import logging
import argparse
import copy
from io import BytesIO

import boto3
import torch
from torch.utils.tensorboard import SummaryWriter
import transformers

from utilities import deconstruct_s3_uri, timing, load_model_and_tokenizer, join


print(f"torch {torch.__version__} transformers: {transformers.__version__}")
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[Evo Training]%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)])

LOG_DIR="/opt/ml/output/tensorboard"
tensorboard_callback = SummaryWriter(log_dir=LOG_DIR)
s3 = boto3.client("s3")


def parse_arguments():
    parser = argparse.ArgumentParser(
        "Use Evo to compute embeddings for the downstream task.")
    parser.add_argument("--data_dir", type=str,
                        default=os.environ.get("SM_CHANNEL_DATA", f"{os.environ['HOME']}/data"),
                        help="Path to dataset.")
    parser.add_argument("--output_s3_prefix_uri", type=str, required=True,
                        help="Location in s3 of the input examples.jsonl and "
                        "this is where the outputs go.")
    parser.add_argument("--model_checkpoint", type=str,
                        default='togethercomputer/evo-1-8k-base',
                        help="Model checkpoint path.")
    parser.add_argument("--augment_datasets", type=int, default=0,
                        help="0 => don't augment; 1 => do augment")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="This is the batch size for the evo-model when creating embeddings.")
    parser.add_argument("--log_level", type=str, default="INFO",
                        help="Log Level.")
    args = parser.parse_args()
    return args


def tokenize_raw_data_batch(raw_data: List[str], tokenizer: Callable) -> torch.tensor:
    tokenized = [tokenizer(dna_seq) for dna_seq in raw_data]
    # print(f"tokenized: {tokenized}")
    return torch.tensor(tokenized, dtype=torch.long)


def do_evo_forward_pass(tokenizer: Callable,
                        evo_model, # StripedHyenaModelForCausalLM
                        batch_size: int,
                        batch_inputs: List[str]): # list of DNA seqs
    """
    Run one batch of DNA sequences thru the Evo model and return the
    batch of embeddings.
    """
    # print(f"do_evo_forward_pass")
    # print(f"batch inputs: {batch_inputs}")
    input_data = tokenize_raw_data_batch(batch_inputs,
                                         lambda dna_seq: tokenizer([dna_seq],
                                                                   add_special_tokens=False)["input_ids"])
    # print(f"input_data: {input_data.shape}")
    input_data = torch.LongTensor(input_data).to("cuda")
    # print(f"input_data: {input_data.shape}")
    input_data = input_data.reshape((batch_size,-1))
    # input_data = input_data.T
    # print(f"input_data: {input_data.shape}")
    outputs = evo_model(input_ids=input_data, output_hidden_states=True)
    logits = outputs.logits.to("cpu")
    # print(f"logits {type(logits)} {logits.shape}")
    # print(f"input_data: {input_data.shape} -> logits {logits.shape}")
    # logits = logits.reshape((logits.shape[0], -1))
    # print(f"logits: {str(logits.tolist())[:200]}")
    return logits


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


def preprocess_training_data(data_dir: str,
                             augment_datasets: bool,
                             batch_size: int,
                             tokenizer: Callable,
                             evo_model,
                             output_s3_prefix_uri: str):
    """
    We use the evo-model to transform each 150-base-pair read into an embedding.
    
    The `data_dir` must contain a file `examples.jsonl`, each line of which
    contains a list of DNA reads (the input to the model) and an impact
    score (the regression output), and other stuff.
    
    The output is put in `output_s3_prefix_uri`.    
    """
    print(f"preprocess_training_data {data_dir}")
    with (Path(data_dir) / "examples.jsonl").open() as f:
        examples = [json.loads(line) for line in f]
    print(f"Loaded {len(examples):,} examples")
    # inputs = [ex["reads"][0] for ex in examples]
    targets = [ex["impact_score"] for ex in examples]
    targets = normalize_targets(targets)
    read_lens = {len(ex["reads"][0]) for ex in examples}
    assert len(read_lens) == 1, read_lens
    # read_len = list(read_lens)[0]

    evo_model.eval()
    with timing("Evo forward passes", num_iterations=len(examples)):
        with torch.no_grad():
            for i, a_read in enumerate(ex["reads"][0] for ex in examples):
                output = do_evo_forward_pass(tokenizer, evo_model, batch_size, a_read)
                examples[i]["embeddings"] = [output]  # for now only one read, but could be more

    target_dict = show_target_distribution(targets)
    if augment_datasets:
        target_2_examples = defaultdict(list)
        for example, target in zip(examples, targets):
            target_2_examples[target].append(example)
        # The distribution in original dataset is:
        # 0.0: 98.4%  0.1: 1.26%  0.2: 0.16%  0.5: 0.2%  1.0: 0.02%
        boosts = {0.1: 100, 0.2: 200, 0.5: 200, 1.0: 750}
        # boosted_outputs = copy.copy(outputs)
        # boosted_targets = copy.copy(targets)
        for target, boost in boosts.items():
            for _ in range(boost*target_dict[target]):
                new_example = choice(target_2_examples[target])
                examples.append(copy.copy(new_example))
        _ = show_target_distribution(normalize_targets([ex["impact_score"]
                                                        for ex in examples]))
    # Upload results back to s3:
    # tensors aren't serializable directly as JSON, they
    # can be indirectly but this is very slow:
    with timing("Uploading embeddings"):
        embeddings = [ex["embeddings"] for ex in examples]
        with BytesIO() as f:
            torch.save(embeddings, f)
            f.seek(0)
            out_bucket, out_path = deconstruct_s3_uri(output_s3_prefix_uri)
            out_path = join("/", out_path, "examples-processed-embeddings.pt")
            # while out_path.endswith("/"): out_path = out_path[:-1]
            # out_path = out_path + "/examples-processed-embeddings.pt"
            s3.upload_fileobj(f, out_bucket, out_path)
            print(f"Uploaded {len(embeddings):,} embeddings to s3://{out_bucket}/{out_path}")
    with timing("Uploading examples"):
        # remove the embeddings as they are way to slow too serialize as JSON,
        # they are instead in the .pt file:
        for example in examples:
            del example["embeddings"]
        example_jsons = [json.dumps(example) for example in examples]
        with BytesIO(("\n".join(example_jsons)).encode("utf-8")) as f:
            out_bucket, out_path = deconstruct_s3_uri(output_s3_prefix_uri)
            out_path = join("/", out_path, "examples-processed.jsonl")
            # while out_path.endswith("/"): out_path = out_path[:-1]
            # out_path = out_path + "/examples-processed.jsonl"
            s3.upload_fileobj(f, out_bucket, out_path)
            print(f"Uploaded {len(examples):,} examples to s3://{out_bucket}/{out_path}")

def main(args):
    model_id = args.model_checkpoint
    print(f"model_id {model_id}")
    model, tokenizer = load_model_and_tokenizer(model_id, True)
    preprocess_training_data(args.data_dir,
                             args.augment_datasets > 0,
                             args.batch_size,
                             tokenizer, model,
                             args.output_s3_prefix_uri)


if __name__ == "__main__" :
    os.system("python --version")
    args = parse_arguments()
    logger.setLevel(args.log_level)
    logger.info("Starting data processing.......")
    main(args)