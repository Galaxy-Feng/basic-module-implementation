import torch
from torch import nn
import tensorflow as tf

# Load and preprocess the Fashion MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(trainImages, trainLabels), (testImages, testLabels) = fashion_mnist.load_data()
trainImages = torch.tensor(trainImages, dtype=torch.float32).div(255)
trainLabels = torch.tensor(trainLabels, dtype=torch.long)
testImages = torch.tensor(testImages, dtype=torch.float32).div(255)
testLabels = torch.tensor(testLabels, dtype=torch.long)
batch_size = 256
train_dataset = torch.utils.data.TensorDataset(trainImages, trainLabels)
test_dataset = torch.utils.data.TensorDataset(testImages, testLabels)
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the model
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

# Initialize weights
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights)

# Define the loss function
loss = nn.CrossEntropyLoss(reduction='none')

# Define the optimizer
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

# Accumulator class for metrics
class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Train one epoch
def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]

# Evaluate accuracy
def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.size(0))
    return metric[0] / metric[1]

# Calculate accuracy
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

# Train the model
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        print(f'Epoch {epoch + 1}, Train Loss: {train_metrics[0]:.4f}, Train Acc: {train_metrics[1]:.4f}, Test Acc: {test_acc:.4f}')

# Train the model for 10 epochs
num_epochs = 10
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)