import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
from model import MNIST_CNN, MNIST_LogisticRegression, Sent140LSTM
from client import Client

class Server:
    def __init__(self, config, train_loaders, test_loader, device):
        self.config = config
        self.train_loaders = train_loaders
        self.test_loader = test_loader
        self.device = device
        self.accuracy_history = []
        self.loss_history = []
        self.loss_test = []
 
        if self.config.dataset == "sent140":
            if self.config.embedding_matrix is None:
                raise RuntimeError("config.embedding_matrix não foi preenchida pelo dataloader.")
            self.global_model = Sent140LSTM(
                embedding_matrix=self.config.embedding_matrix,
                pad_idx=self.config.pad_idx,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                num_classes=self.config.num_classes,
            ).to(self.device)
        else:
            self.global_model = MNIST_LogisticRegression().to(self.device)

    def select_clients(self):
        K = len(self.train_loaders)
        num_selected = max(1, int(self.config.frac * K))
        return random.sample(range(K), num_selected)

    def train_clients(self, selected_clients):
        local_models = []
        local_losses = []
        local_sample_counts = []
        # Snapshot dos parâmetros globais — SEM deepcopy, só detach
        global_params_snapshot = [p.detach().clone() 
                                   for p in self.global_model.parameters()]
    
        num_selected = len(selected_clients)
        num_stragglers = int(round(self.config.straggler_rate * num_selected))
        stragglers = set(random.sample(selected_clients, num_stragglers))
    
        for idx in selected_clients:
            client_epochs = self.config.local_epochs
    
            if idx in stragglers:
                client_epochs = random.randint(1, self.config.local_epochs)
                if client_epochs < self.config.local_epochs and self.config.algorithm == "fedavg":
                    continue        

            client_config = copy.deepcopy(self.config)
            client_config.local_epochs = client_epochs
            
            local_model = copy.deepcopy(self.global_model)
            client = Client(local_model, self.train_loaders[idx], client_config, self.device, idx)
            trained_model, loss, nk = client.train(global_model=global_params_snapshot)
            local_models.append(trained_model)
            local_losses.append(loss)
            local_sample_counts.append(nk)
            del client
        return local_models, local_losses, local_sample_counts

    def avg_grads(self, global_model, local_models, weights):
        # Average model parameters (FedAvg) - Agora com pesos (nk/n)
        global_dict = global_model.state_dict()
        
        # Calcula as proporções de cada cliente (nk / n_total)
        total_samples = sum(weights)
        proportions = torch.tensor([w / total_samples for w in weights], dtype=torch.float32)
        
        for key in global_dict.keys():
            # SEU CÓDIGO: empilha os tensores de todos os modelos
            stacked = torch.stack([lm.state_dict()[key].float() for lm in local_models], dim=0)
            
            # NOVO: Transforma o vetor de proporções para ter as mesmas dimensões do 'stacked'
            # Ex: Se stacked é (10 clientes, 256, 256), isso transforma proportions em (10, 1, 1)
            props_reshaped = proportions.view(-1, *[1]*(stacked.dim() - 1)).to(stacked.device)
            
            # Substitui o seu stacked.mean(dim=0) pela soma ponderada:
            global_dict[key] = (stacked * props_reshaped).sum(dim=0)
            
        global_model.load_state_dict(global_dict)
        return global_model
        
    def evaluate_train_loss(self):
        """Avalia f(w) sem contaminar o RNG do treinamento."""
        # Salva TODOS os estados de RNG antes de avaliar
        rng_state_torch = torch.get_rng_state()
        rng_state_np = np.random.get_state()
        if torch.cuda.is_available():
            rng_state_cuda = torch.cuda.get_rng_state()
    
        self.global_model.eval()
        total_loss = 0.0
        total_samples = 0
    
        with torch.no_grad():
            for idx in range(len(self.train_loaders)):
                for data, target in self.train_loaders[idx]:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.global_model(data)
                    loss = F.cross_entropy(output, target, reduction='sum')
                    total_loss += loss.item()
                    total_samples += target.size(0)
    
        # Restaura TODOS os estados — treinamento continua como se evaluate não existisse
        torch.set_rng_state(rng_state_torch)
        np.random.set_state(rng_state_np)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(rng_state_cuda)
    
        return total_loss / total_samples   
        
    def evaluate(self):
        self.global_model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                
                loss = F.cross_entropy(output, target, reduction='sum')
                total_loss += loss.item()

                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
                
        avg_loss = total_loss / total
        acc = 100.0 * correct / total
        return acc, avg_loss

    def run(self):
        for r in range(1, self.config.num_rounds + 1):
            selected = self.select_clients()
            local_models, losses, counts = self.train_clients(selected)
            if len(local_models) > 0:
                self.global_model = self.avg_grads(self.global_model, local_models, counts)
            else:
                print(f"Rodada {r} ignorada: Todos os clientes falharam (drop).")
                
            # Sem argumento — usa self.train_loaders internamente
            train_loss = self.evaluate_train_loss()
            self.loss_history.append(train_loss)
            acc, test_loss = self.evaluate()
            self.accuracy_history.append(acc)
            self.loss_test.append(test_loss)
           
            print(f"Round {r:3d} | Global Test Accuracy: {acc:.2f}% | Training Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
            #if acc >= 99:
              #  print(f"round reached {r:3d} | batch size {self.config.batch_size} | frac {self.config.frac}") 
               # break