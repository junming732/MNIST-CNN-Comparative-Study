"""
Multi-Layer Perceptron model for MNIST classification
"""

import torch.nn as nn

class MLP(nn.Module):
    """Multi-Layer Perceptron model for MNIST classification"""
    
    def __init__(self, input_dim=784, hidden_dim=280, output_dim=10):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.hidden_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x