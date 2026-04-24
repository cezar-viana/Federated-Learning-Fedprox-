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
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(-1, 784)
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
        x = self.embedding(x)
        outputs, _ = self.lstm(x)
        last_output = outputs[:, -1, :]
        logits = self.classifier(last_output)
        return logits
