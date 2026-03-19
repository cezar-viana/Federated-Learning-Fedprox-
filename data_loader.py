# data_loader.py
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def load_mnist_partitions(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    num_clients = config.num_clients
    train_loaders = []

    if config.iid:
        # -------------------------------------------------------------------
        # LÓGICA IID (Dados misturados e divididos igualmente)
        # -------------------------------------------------------------------
        print("Preparando dados em formato IID...")
        data_per_client = len(train_dataset) // num_clients
        
        # Opcional mas recomendado: embaralhar os índices gerais antes de dividir 
        # para garantir uma distribuição IID perfeita
        all_indices = np.random.permutation(len(train_dataset))

        for i in range(num_clients):
            start = i * data_per_client
            end = start + data_per_client if i < num_clients - 1 else len(train_dataset)
            
            client_indices = all_indices[start:end]
            subset = Subset(train_dataset, client_indices.tolist())        
            
            if config.batch_size is None:
                batch_size = len(subset)
                shuffle = False
            else:
                batch_size = config.batch_size
                shuffle = True
                
            loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle)    
            train_loaders.append(loader)

    else:
        # -------------------------------------------------------------------
        # LÓGICA NON-IID (FedProx: 2 classes por cliente + Power Law)
        # -------------------------------------------------------------------
        print("Preparando dados em formato Non-IID (FedProx / Power Law)...")
        labels = train_dataset.targets.numpy()
        num_classes = 10
        
        # Agrupa os índices originais por classe
        class_indices = {i: np.where(labels == i)[0].tolist() for i in range(num_classes)}
        
        # Embaralha os índices dentro de cada classe
        for i in range(num_classes):
            np.random.shuffle(class_indices[i])

        # Gera proporções seguindo uma distribuição Power Law
        # O parâmetro 'a' < 1 garante que a maioria terá valores baixos (muitos clientes com poucos dados)
        power_law_props = np.random.power(a=0.5, size=num_clients)
        power_law_props = power_law_props / power_law_props.sum() # Normaliza para somar 1.0
        
        # Calcula quantas amostras cada cliente terá baseado na proporção (mínimo de 10 amostras por cliente)
        client_sizes = (power_law_props * len(train_dataset)).astype(int)
        client_sizes = np.clip(client_sizes, a_min=10, a_max=None)

        for i in range(num_clients):
            # 1. Escolhe 2 dígitos distintos aleatórios para este cliente
            client_classes = np.random.choice(num_classes, 2, replace=False)
            
            # 2. Divide o tamanho alvo do cliente entre as duas classes
            size = client_sizes[i]
            size_per_class = size // 2
            
            client_indices = []
            
            for c in client_classes:
                # Se ainda tivermos amostras não utilizadas dessa classe, pegamos elas
                if len(class_indices[c]) >= size_per_class:
                    selected = class_indices[c][:size_per_class]
                    class_indices[c] = class_indices[c][size_per_class:] # Remove as usadas do pool
                else:
                    # Se esgotarmos as amostras únicas dessa classe, amostramos com reposição 
                    # de toda a base daquela classe apenas para manter o desbalanceamento da Power Law
                    all_class_c_indices = np.where(labels == c)[0]
                    selected = np.random.choice(all_class_c_indices, size_per_class, replace=True).tolist()
                    
                client_indices.extend(selected)
                
            np.random.shuffle(client_indices)
            
            # Cria o subset do cliente
            subset = Subset(train_dataset, client_indices)        
            
            if config.batch_size is None:
                batch_size = len(subset) # Cuidado aqui, alguns clientes podem ter arrays muito grandes
                shuffle = False
            else:
                batch_size = config.batch_size
                shuffle = True
                
            loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle)    
            train_loaders.append(loader)

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loaders, test_loader