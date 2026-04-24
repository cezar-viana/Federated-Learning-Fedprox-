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
        """Perform local training, showing epoch and batch progress."""
        self.model.train()
        epochs = 1 if self.config.algorithm == "fedsgd" else self.config.local_epochs
        final_epoch_loss = 0.0

        for epoch in range(1, epochs + 1):
            batch_losses = []
            # iterate over batches with tqdm
            for data, target in self.train_loader:     
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()

                # call step with correct signature
                if self.config.algorithm == "fedavg":
                    self.optimizer.step()
                elif self.config.algorithm == "fedprox":
                    self.optimizer.step(global_params=global_model)
                elif self.config.algorithm == "feddane":
                    self.optimizer.step(
                        global_params=global_model,
                        global_gradients=global_grads
                    )
                elif self.config.algorithm == "fedsgd":
                    self.optimizer.step(global_gradients=global_grads)
                    
                batch_losses.append(loss.item())
                
            final_epoch_loss = sum(batch_losses) / len(batch_losses)
        #print(f"final loss = {final_epoch_loss}")
            
        num_samples = len(self.train_loader.dataset)

        return self.model, final_epoch_loss, num_samples