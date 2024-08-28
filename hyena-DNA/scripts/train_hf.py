
import sys
import os
from os.path import expanduser, join

import torch
from torch import optim
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from species_dataset import SpeciesDataset

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import timedelta

import logging
from torch.utils.tensorboard import SummaryWriter

import argparse
from accelerate import Accelerator

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[HyenaDNA Training]%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Set tensorboard 
LOG_DIR="/opt/ml/output/tensorboard"
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


def train(model, train_loader, optimizer, lr_scheduler, device, log_interval, epoch, train_sampler):
    model.train()
    train_sampler.set_epoch(epoch)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        logger.info(f"Data device: {data.device}, Target device: {target.device}, Model device: {next(model.parameters()).device}")

        
        output = model(input_ids=data, labels=target)
        loss = output.loss
        accelerator.backward(loss)
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        loss_tensor = torch.tensor([loss.item()]).to(device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_tensor /= dist.get_world_size()
        
        global_loss = loss_tensor.item()
        global_perplexity = calculate_perplexity(global_loss)
        
        if dist.get_rank() == 0:
            tensorboard_callback.add_scalar('Training Loss', global_loss, epoch * len(train_loader) + batch_idx)
            tensorboard_callback.add_scalar('Training Perplexity', global_perplexity, epoch * len(train_loader) + batch_idx)
            
            if batch_idx % log_interval == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)], Train Loss: {:.6f}, Train Perplexity: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), global_loss, global_perplexity))

            

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

    # Convert total loss and total items to tensors for all_reduce operation
    total_loss_tensor = torch.tensor([total_loss]).to(device)
    total_items_tensor = torch.tensor([total_items]).to(device)
    
    # Use all_reduce to sum these tensors across all processes
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_items_tensor, op=dist.ReduceOp.SUM)

    global_avg_loss = total_loss_tensor.item() / total_items_tensor.item()
    global_avg_perplexity = calculate_perplexity(global_avg_loss)

    if dist.get_rank() == 0:
        tensorboard_callback.add_scalar('Evaluation Loss', global_avg_loss, epoch)
        tensorboard_callback.add_scalar('Evaluation Perplexity', global_avg_perplexity, epoch)
        
        logger.info('\n Epoch {}. Eval Average Loss: {:.4f}, Eval Perplexity: {:.6f}\n'.format(
            epoch, global_avg_loss, global_avg_perplexity))


def save_model(model, model_dir):
    model = model.module if hasattr(model, "module") else model
    os.makedirs(model_dir, exist_ok=True)
    checkpoint = {"state_dict": model.state_dict()}
    path = f"{model_dir}/checkpoint.pt"
    torch.save(checkpoint, path)


def init_distributed_training():
    world_size = int(os.environ["WORLD_SIZE"])
    global_rank = int(os.environ["RANK"])    
    local_rank = int(os.environ['LOCAL_RANK'])

    logger.info("Distribution setup is done with world size [{}]".format(world_size))

    torch.cuda.set_device(local_rank)
    
    dist.init_process_group(backend="nccl", world_size=world_size, rank=global_rank, init_method="env://", timeout=timedelta(seconds=120))

    device = torch.device(f"cuda:{local_rank}")

    return local_rank, device

def cleanup():
    dist.destroy_process_group()
    
def main(args):

    local_rank, device = init_distributed_training()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, trust_remote_code=True)
    
    train_dataset, test_dataset = get_dataset(tokenizer, args.max_length)
    train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    test_sampler = DistributedSampler(test_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, shuffle=False)

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
    
    model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    
    for epoch in range(args.epochs):
        train(model, train_dataloader, optimizer, lr_scheduler, device, args.log_interval, epoch, train_sampler)
        eval(model, test_dataloader, device, epoch)

    dist.barrier()

    if dist.get_rank() == 0:
        save_model(model, args.model_dir)

    cleanup()


if __name__ == "__main__" :

    args = parse_arguments()
    logger.setLevel(args.log_level)
    logger.info("Starting training.......")
    main(args)