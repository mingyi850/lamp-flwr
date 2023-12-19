from collections import OrderedDict
from typing import Dict, List, Tuple

import os
import numpy as np
import torch

import secagg
import flwr as fl
import torch.nn as nn

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from datasets import load_metric

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SecAggClient(fl.client.NumPyClient):
    def __init__(
        self,
        model: nn.Module,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        num_examples: Dict,
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples

    def get_parameters(self, config) -> List[np.ndarray]:
        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        secagg.trainOne(self.model, self.trainloader, epochs=1, device=DEVICE)
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = secagg.test(self.model, self.testloader, device=DEVICE)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}

def main() -> None:
    """Load data, start SecAggClient."""
    devicestr = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(devicestr)
    # Load model and data
    model = AutoModelForSequenceClassification.from_pretrained("huawei-noah/TinyBERT_General_6L_768D")
    model.to(DEVICE)
    trainloader, testloader, num_examples = secagg.get_tokenized_datasets(1)
    # Start client
    master_addr = os.getenv('MASTER_ADDR', '0.0.0.0')
    master_port = os.getenv('MASTER_PORT', '8080')
    print("Connecting to server at:", f"{master_addr}:{master_port}")
    client = SecAggClient(model, trainloader, testloader, num_examples)
    fl.client.start_numpy_client(server_address=f"{master_addr}:{master_port}", client=client)


if __name__ == "__main__":
    main()