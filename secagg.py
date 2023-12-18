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

class Net(nn.Module):

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

DATA_ROOT = "~/data/cifar-10"


def load_data() -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict]:
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10(DATA_ROOT, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    testset = CIFAR10(DATA_ROOT, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
    num_examples = {"trainset" : len(trainset), "testset" : len(testset)}
    return trainloader, testloader, num_examples

def train(
    net: Net,
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
            loss += loss.item()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    
def trainOne(
net: Net,
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
    net: Net,
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

def main():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Centralized PyTorch training")
    print("Load data")
    trainloader, testloader, _ = load_data()
    print("Start training")
    net=Net().to(DEVICE)
    train(net=net, trainloader=trainloader, epochs=2, device=DEVICE)
    print("Evaluate model")
    loss, accuracy = test(net=net, testloader=testloader, device=DEVICE)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)


if __name__ == "__main__":
    main()