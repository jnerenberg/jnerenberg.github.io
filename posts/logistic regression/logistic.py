import torch
import torch.nn as nn

class LinearModel:
    def __init__(self):
        self.w = None
    
    def score(self, X):
        return X @ self.w
    
    def predict(self, X):
        return self.score(X) > 0
    
class LogisticRegression(nn.Module):

    # Initialize the model with the input dimension
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        # gradient
        self.linear.weight.data.fill_(0)
        self.linear.bias.data.fill_(0)

    def sigmoid(self, z):
        '''
        Sigmoid activation function
        '''
        return 1 / (1 + torch.exp(-z))
    
    def forward(self, X):
        '''
        Forward pass of the model
        '''
        z = self.linear(X)
        return self.sigmoid(z)
    
    def loss(self, y_pred, y):
        '''
        Binary cross-entropy loss
        '''
        return (1/y_pred.size(0)) * ( (-y_pred.log()*y) + (-(1-y_pred).log()*(1-y)) ).sum()
    
    def grad(self):
        '''
        Compute the gradients of the loss w.r.t the model parameters
        '''
        return self.linear.weight.grad, self.linear.bias.grad
    
    def gradient(self):
        '''
        Compute the gradients of the loss w.r.t the model parameters, but with the full name
        '''
        return self.linear.weight.grad, self.linear.bias.grad
    
class GradientDescentOptimizer:
    def __init__(self, model, learning_rate, momentum=0):
        '''
        Initialize the optimizer with the model, learning rate and momentum
        '''
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = 0
        
    def step(self):
        '''
        Perform a single optimization step
        '''
        
        # Compute the gradients
        gradients = self.model.grad()

        # Update the velocity
        self.velocity = self.momentum * self.velocity + self.learning_rate * gradients

        # Update the model parameters
        for param in self.model.parameters():
            param.data -= self.velocity


