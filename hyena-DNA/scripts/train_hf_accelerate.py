
import sys
import os
from os.path import join

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from species_dataset import SpeciesDataset

import torch.distributed as dist

import logging
from torch.utils.tensorboard import SummaryWriter

import argparse
from accelerate import Accelerator

import preprocess_data as data_proc

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[HyenaDNA Training]%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Set tensorboard 
LOG_DIR="/opt/ml/output/tensorboard"
#LOG_DIR="/home/ubuntu/tmp/tb"
tensorboard_callback = SummaryWriter(log_dir=LOG_DIR)

accelerator = Accelerator()

def parse_arguments():
    parser = argparse.ArgumentParser("Arguements for pre-training HyenaDNA model.")
   
    parser.add_argument("--species", nargs="+", required=True, help="Species types that we train the model on.")
    parser.add_argument("--data_dir", type=str, default=os.environ["SM_CHANNEL_DATA"], help="Path to dataset.")
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"], help="Path to model output folder.")
    
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs.")
    parser.add_argument("--model_checkpoint", type=str, default='LongSafari/hyenadna-small-32k-seqlen-hf', help="Model checkpoint path.")
    parser.add_argument("--max_length", type=int, default=32_000, help="Maximum sequence length.")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=6e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight Decay.")
    
    parser.add_argument("--log_level", type=str, default="INFO", help="Log Level.")
    parser.add_argument("--log_interval", type=int, default=1, help="Log Interval.")

    args = parser.parse_args()
    return args


def get_dataset(tokenizer, max_length):

    train_dataset = SpeciesDataset(
           species=args.species,
           species_dir=join(args.data_dir),
           split="train",
           max_length=max_length,
           total_size=10000,
           pad_max_length=None,
           tokenizer=tokenizer,
           tokenizer_name="char",
           add_eos=False,
           rc_aug=False,
           return_augs=False,
           chromosome_weights='uniform',
           species_weights='uniform',
           task='next_token_pred',
           remove_tail_ends=False,
           cutoff_train=0.1,
           cutoff_test=0.2)
    
    test_dataset = SpeciesDataset(
           species=args.species,
           species_dir=join(args.data_dir),
           split="test",
           max_length=max_length,
           total_size=1000,
           pad_max_length=None,
           tokenizer=tokenizer,
           tokenizer_name="char",
           add_eos=False,
           rc_aug=False,
           return_augs=False,
           chromosome_weights='uniform',
           species_weights='uniform',
           task='next_token_pred',
           remove_tail_ends=False,
           cutoff_train=0.1,
           cutoff_test=0.2)

    return train_dataset, test_dataset


def train(model, train_loader, optimizer, lr_scheduler, log_interval, epoch):
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        outputs = model(input_ids=data, labels=target)
        loss = outputs.loss
        accelerator.backward(loss)
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        local_loss = loss.item()
        local_perplexity = calculate_perplexity(local_loss)

        if accelerator.is_main_process:
            tensorboard_callback.add_scalar('Training Loss', local_loss, epoch * len(train_loader) + batch_idx)
            tensorboard_callback.add_scalar('Training Perplexity', local_perplexity, epoch * len(train_loader) + batch_idx)

            if batch_idx % log_interval == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)], Train Loss: {:.6f}, Train Perplexity: {:.6f}'.format(
                            epoch, batch_idx * len(data), len(train_loader.dataset),
                            100. * batch_idx / len(train_loader), local_loss, local_perplexity))


def calculate_perplexity(avg_loss):
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity.item()


def eval(model, test_dataloader, epoch):
    model.eval()
    total_loss = 0
    total_items = 0

    with torch.no_grad() :
        for batch_idx, (data, target) in enumerate(test_dataloader):
            output = model(input_ids=data, labels=target)
            loss = output.loss

            total_loss += loss.item() * data.size(0)
            total_items += data.size(0)


    avg_loss = total_loss / total_items
    avg_perplexity = calculate_perplexity(avg_loss)

    if accelerator.is_main_process:
        tensorboard_callback.add_scalar('Evaluation Loss', avg_loss, epoch)
        tensorboard_callback.add_scalar('Evaluation Perplexity', avg_perplexity, epoch)
        
        logger.info('\n Epoch {}. Eval Average Loss: {:.4f}, Eval Perplexity: {:.6f}\n'.format(
            epoch, avg_loss, avg_perplexity))


def save_model(model, model_dir):
    model = model.module if hasattr(model, "module") else model
    os.makedirs(model_dir, exist_ok=True)
    checkpoint = {"state_dict": model.state_dict()}
    path = f"{model_dir}/checkpoint.pt"
    torch.save(checkpoint, path)


def init_distributed_training():
    if accelerator.is_local_main_process :
        logger.info(f"Accelerate State: {accelerator.state}")
        logger.info(f"Is Main Process: {accelerator.is_main_process}")
        logger.info(f"Local Process Index: {accelerator.local_process_index}")
        logger.info(f"Device: {accelerator.device}")
        logger.info(f"Number of Processes: {accelerator.num_processes}")

    
def main(args):

    if accelerator.is_local_main_process:
        data_proc.preprocess_data_for_hyenaDNA(args.data_dir, args.species[0])

    accelerator.wait_for_everyone()

    init_distributed_training()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, trust_remote_code=True)
    
    train_dataset, test_dataset = get_dataset(tokenizer, args.max_length)
    print("Train data set size " + str(len(train_dataset)))
    print("Test data set size " + str(len(test_dataset)))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = AutoModelForCausalLM.from_pretrained(args.model_checkpoint, torch_dtype=torch.bfloat16, trust_remote_code=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=(len(train_dataloader) * args.epochs)/args.batch_size)
    
    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(model, optimizer, train_dataloader, test_dataloader)

    
    for epoch in range(args.epochs):
        train(model, train_dataloader, optimizer, lr_scheduler, args.log_interval, epoch)
        eval(model, test_dataloader, epoch)

    if accelerator.is_main_process:
        save_model(model, args.model_dir)


if __name__ == "__main__" :

    args = parse_arguments()
    logger.setLevel(args.log_level)
    logger.info("Starting training.......")
    main(args)