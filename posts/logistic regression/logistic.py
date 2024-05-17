import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        return self.sigmoid(self.linear(x))
    
class GradientDescentOptimizer:
    def __init__(self, model, learning_rate):
        self.model = model
        self.learning_rate = learning_rate
        
    def step(self):
        for param in self.model.parameters():
            param.data -= self.learning_rate * param.grad.data
            param.grad.data.zero_()

class NewtonOptimizer:
    def __init__(self, model):
        self.model = model
        
    def step(self):
        x = self.model.linear.weight.data
        y = self.model.linear.bias.data
        z = self.model.sigmoid(x)
        z = z * (1 - z)
        z = z.view(-1)
        z = z.unsqueeze(1)
        z = z.expand(-1, x.size(1))
        z = z.unsqueeze(2)
        z = z.expand(-1, -1, x.size(1))
        hessian = x.t().unsqueeze(0).expand(x.size(0), -1, -1)
        hessian = hessian * z
        hessian = hessian.bmm(x.unsqueeze(2))
        hessian = hessian.squeeze(2)
        hessian = hessian.inverse()
        gradient = self.model.linear.weight.grad.data
        x.data -= hessian.bmm(gradient.unsqueeze(2)).squeeze(2)
        gradient = self.model.linear.bias.grad.data
        y.data -= hessian.bmm(gradient.unsqueeze(2)).squeeze(2)

def train(model, optimizer, criterion, x, y, epochs):
    for epoch in range(epochs):
        optimizer.model.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        print('Epoch: {0}, Loss: {1}'.format(epoch, loss.item()))

def main():
    np.random.seed(0)
    x = np.random.randn(100, 2)
    y = np.random.randint(0, 2, (100, 1))
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    model = LogisticRegression(2)
    criterion = nn.BCELoss()
    optimizer = GradientDescentOptimizer(model, 0.1)
    train(model, optimizer, criterion, x, y, 100)
    optimizer = NewtonOptimizer(model)
    train(model, optimizer, criterion, x, y, 100)

if __name__ == '__main__':
    main()
