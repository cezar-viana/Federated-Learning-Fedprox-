# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
    
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
    
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MNIST_LogisticRegression(nn.Module):
    def __init__(self):
        super(MNIST_LogisticRegression, self).__init__()
        # A entrada é 28*28 = 784 pixels. A saída são as 10 classes (dígitos 0 a 9)
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        # Transforma o tensor da imagem (Batch, 1, 28, 28) em um vetor plano (Batch, 784)
        x = x.view(-1, 784)
        
        # Aplica a transformação linear
        out = self.linear(x)
        return out

class Sent140LSTM(nn.Module):
    def __init__(
        self,
        embedding_matrix,
        pad_idx: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 2,
    ):
        super().__init__()

        emb = torch.tensor(embedding_matrix, dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(
            emb,
            freeze=False,
            padding_idx=pad_idx,
        )
        self.lstm = nn.LSTM(
            input_size=emb.shape[1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [batch, seq_len]
        x = self.embedding(x)           # [B, T, 300]
        _, (h_n, _) = self.lstm(x)      # h_n: [num_layers, B, H]
        last_hidden = h_n[-1]           # [B, H]
        logits = self.classifier(last_hidden)
        return logits