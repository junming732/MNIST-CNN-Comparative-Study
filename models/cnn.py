"""
Convolutional Neural Network model for MNIST classification
"""

import torch.nn as nn

class SimpleCNN(nn.Module):
    """Convolutional Neural Network model for MNIST classification"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),  # 1x28x28 -> 8x28x28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x14x14
            
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),  # 8x14x14 -> 16x14x14
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x7x7
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 16x7x7 -> 32x7x7
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*7*7, num_classes)
        )
        
    def forward(self, x):
        # Reshape if needed (for flat input)
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        x = self.features(x)
        x = self.classifier(x)
        return x