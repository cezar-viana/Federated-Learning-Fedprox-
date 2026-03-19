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
