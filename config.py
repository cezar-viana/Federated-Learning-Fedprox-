class FLConfig:
    def __init__(self, algorithm: str = "fedavg", num_clients: int = 1000, local_epoch: int = 5, batch_size: int | None = 10, iid: bool = True, mu:  float = 0.0, straggler_rate: float = 0.0):
        # Federated learning settings
        self.algorithm = algorithm
        self.num_clients = num_clients
        self.num_rounds = 100                # Total communication rounds
        self.local_epochs = local_epoch          # Local epochs per client
        self.frac = 0.01                         # Fraction of clients each round
        self.local_lr = 0.03                     # Learning rate for local updates
        self.global_lr = 1.0                    # (Not used in FedAvg) global learning rate
        self.batch_size = batch_size          # Batch size for training
        self.use_cuda = True                    # Use GPU if available
        self.iid = iid
        self.mu = mu
        self.straggler_rate = straggler_rate

        # Where to save the global model
        self.save_path = "models/global_{algorithm}.pth"
