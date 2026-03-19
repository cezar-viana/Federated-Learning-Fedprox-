from tqdm import trange, tqdm
import random
import torch
import torch.nn.functional as F
from optimizers import (
    FedAvgOptimizer,
    FedProxOptimizer,
    FedDANEOptimizer,
    FedSGDOptimizer,
)

class Client:
    """Encapsulates a single client’s local training logic, with progress bars."""
    def __init__(self, model, train_loader, config, device, client_id):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.config       = config
        self.device       = device
        self.client_id    = client_id


        # Map algorithm name to optimizer class
        optimizer_map = {
            "fedavg":  FedAvgOptimizer,
            "fedprox": FedProxOptimizer,
            "feddane": FedDANEOptimizer,
            "fedsgd":  FedSGDOptimizer,
        }
        optimizer_cls = optimizer_map[config.algorithm]

        # Instantiate optimizer (pass mu for prox/DANE)
        if config.algorithm in ("fedprox", "feddane"):
            self.optimizer = optimizer_cls(
                self.model.parameters(),
                lr=config.local_lr,
                mu=config.mu
            )
        else:
            self.optimizer = optimizer_cls(
                self.model.parameters(),
                lr=config.local_lr
            )

    def train(self, global_model=None, global_grads=None):
        # 1. Avaliar o loss do modelo global ANTES de treinar (para o gráfico)
        global_model.eval()
        round_loss = 0.0
        with torch.no_grad():
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = global_model(data)
                round_loss += F.cross_entropy(output, target, reduction='sum').item()
                
        num_samples = len(self.train_loader.dataset)
        client_global_loss = round_loss / num_samples
        #print(client_global_loss)
        # 2. Treino Local
        self.model.train()
        epochs = 1 if self.config.algorithm == "fedsgd" else self.config.local_epochs

        for epoch in range(1, epochs + 1):
            for data, target in self.train_loader:     
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()

                # O seu código original de steps continua aqui
                if self.config.algorithm == "fedavg":
                    self.optimizer.step()
                elif self.config.algorithm == "fedprox":
                    self.optimizer.step(global_params=global_model)
                elif self.config.algorithm == "feddane":
                    self.optimizer.step(global_params=global_model, global_gradients=global_grads)
                elif self.config.algorithm == "fedsgd":
                    self.optimizer.step(global_gradients=global_grads)
                    
        # Agora retorna o loss global calculado lá em cima!
        return self.model, client_global_loss, num_samples
