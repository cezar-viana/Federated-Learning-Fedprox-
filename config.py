# config.py
class FLConfig:
    def __init__(
        self,
        dataset: str = "sent140",
        algorithm: str = "fedavg",
        num_clients: int | None = None,
        local_epoch: int = 20,
        batch_size: int = 10,
        iid: bool = False,
        mu: float | None = None,
        straggler_rate: float = 0.0,
    ):
        self.dataset = dataset.lower()
        self.algorithm = algorithm.lower()

        self.use_cuda = True
        self.iid = iid
        self.straggler_rate = straggler_rate

        # defaults gerais
        self.num_rounds = 200
        self.local_epochs = local_epoch
        self.batch_size = batch_size
        self.global_lr = 1.0
        self.save_path = "models/global_{algorithm}.pth"
        self.clients_per_round = 10

        # campos preenchidos pelo dataloader/modelo
        self.embedding_matrix = None
        self.pad_idx = 0
        self.vocab_size = None

        if self.dataset == "sent140":
            # paper
            self.num_clients = 772 if num_clients is None else num_clients
            self.frac = self.clients_per_round / self.num_clients
            self.num_rounds = 800
            self.local_epochs = local_epoch
            self.batch_size = 10
            self.local_lr = 0.3

            # modelo
            self.seq_len = 25
            self.num_classes = 2
            self.embedding_dim = 300
            self.hidden_size = 256   # paper
            self.num_layers = 2

            # dados
            self.leaf_root = "./leaf/data/sent140/data"
            self.glove_path = "./embeddings/glove.6B.300d.txt"

            # μ do paper para Sent140 na Figura 1
            if mu is None:
                self.mu = 0.01 if self.algorithm == "fedprox" else 0.0
            else:
                self.mu = mu
        else:
            # fallback para os seus outros experimentos
            self.num_clients = 200 if num_clients is None else num_clients
            self.frac = 0.05
            self.local_lr = 0.003
            self.mu = 0.0 if mu is None else mu
