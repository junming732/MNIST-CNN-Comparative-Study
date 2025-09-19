"""
Data loading and preprocessing utilities for MNIST dataset
"""

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

class MNISTDataLoader:
    """Handles loading and preprocessing of MNIST dataset"""
    
    def __init__(self, data_path=None):
        self.data_path = data_path
        
    def load_data(self):
        """
        Load MNIST data from PNG files or use torchvision datasets as fallback
        
        Returns:
            X_train, Y_train, X_test, Y_test: Tensors containing training and test data
        """
        try:
            # Try to load from custom path
            if self.data_path:
                return self._load_from_custom_path()
            else:
                # Fallback to torchvision datasets
                return self._load_from_torchvision()
        except Exception as e:
            print(f"Error loading data: {e}. Using torchvision fallback.")
            return self._load_from_torchvision()
    
    def _load_from_torchvision(self):
        """Load MNIST using torchvision datasets"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Download and load training data
        train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
        # Convert to tensors
        X_train, Y_train = train_data.data.float() / 255.0, train_data.targets
        X_test, Y_test = test_data.data.float() / 255.0, test_data.targets
        
        # Reshape and one-hot encode labels
        X_train = X_train.reshape(-1, 784)
        X_test = X_test.reshape(-1, 784)
        Y_train = F.one_hot(Y_train, num_classes=10).float()
        Y_test = F.one_hot(Y_test, num_classes=10).float()
        
        return X_train, Y_train, X_test, Y_test

    def _load_from_custom_path(self):
        """Load MNIST data from custom file path (to be implemented)"""
        # This would contain your custom loading logic from PNG files
        raise NotImplementedError("Custom path loading not implemented in this example")