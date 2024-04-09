
import sys
import os
from os.path import expanduser, join

import torch
from torch import optim
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from species_dataset import SpeciesDataset

import logging
import argparse

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[HyenaDNA Training]%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

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


def train(model, train_loader, optimizer, lr_scheduler, device, log_interval, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        output = model(input_ids=data, labels=target)
        loss = output.loss
        loss.backward()
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        if batch_idx % log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)], Train Loss: {:.6f}, Train perplexity: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), calculate_perplexity(loss.item())))
            

def calculate_perplexity(avg_loss):
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity.item()


def eval(model, test_dataloader, device, epoch):
    model.eval()
    total_loss = 0
    total_items = 0
    with torch.no_grad() :
        for batch_idx, (data, target) in enumerate(test_dataloader):
            data, target = data.to(device), target.to(device)
            output = model(input_ids=data, labels=target)
            loss = output.loss
            total_loss += loss.item() * data.size(0)
            total_items += data.size(0)
    avg_loss = total_loss/total_items
    logger.info('\n Epoch {}. Eval Average Loss: {:.4f}, Eval perplexity: {:.6f}\n'.format(epoch,
        avg_loss, calculate_perplexity(avg_loss)) )


def save_model(model, model_dir):
    model = model.module if hasattr(model, "module") else model
    os.makedirs(model_dir, exist_ok=True)
    checkpoint = {"state_dict": model.state_dict()}
    path = f"{model_dir}/checkpoint.pt"
    torch.save(checkpoint, path)


def main(args):
   
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, trust_remote_code=True)
    
    train_dataset, test_dataset = get_dataset(tokenizer, args.max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_checkpoint, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        trust_remote_code=True
    )

    #Fix Me to show only on rank 1
    logger.debug(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * args.epochs)/args.batch_size
    )
    
    # Fix Me
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    for epoch in range(args.epochs):
        train(model, train_dataloader, optimizer, lr_scheduler, device, args.log_interval, epoch)
        eval(model, test_dataloader, device, epoch)
    
    save_model(model, args.model_dir)


if __name__ == "__main__" :

    args = parse_arguments()
    logger.setLevel(args.log_level)
    logger.info("Starting training.......")
    main(args)