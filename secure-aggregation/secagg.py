import tokenize
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import random
from torch import Tensor
from torchvision.datasets import CIFAR10
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer
from datasets import load_dataset
    
def trainOne(
net: nn.Module,
trainloader: torch.utils.data.DataLoader,
epochs: int,
device: torch.device,
) -> None:
    """Train the network."""
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")
    # Train the network
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            data = {k: v.to(device) for k, v in data.items()}
            outputs = net(**data)
            logits = outputs.logits

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            predictions = torch.argmax(logits, dim=1)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
            break #only perform one sample of training


def test(
    net: nn.Module,
    testloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            data = {k: v.to(device) for k, v in data.items()}
            outputs = net(**data)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            loss = outputs.loss
            total_loss += loss.item()
            total += 1
    accuracy = 0# correct / total
    return loss, accuracy

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
tokenizer.model_max_length = 512

def tokenize_function(examples):
    seq_key = 'sentence'
    return tokenizer(examples[seq_key], truncation=True)


def get_tokenized_datasets(batch_size):
    datasets = load_dataset('glue', 'cola')
    tokenized_datasets = datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(['idx', 'sentence'])
    tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
    tokenized_datasets.set_format('torch')
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataset = tokenized_datasets['train']
    eval_dataset = tokenized_datasets['validation']

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    eval_loader = DataLoader(eval_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    num_examples = {"trainset" : len(train_dataset), "testset" : len(eval_dataset)}
    return (train_loader, eval_loader, num_examples)