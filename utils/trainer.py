"""
Training and evaluation utilities for neural network models
"""

import time
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class CNNTrainer:
    """Handles training and evaluation of models"""
    
    def __init__(self, model, device=None):
        self.model = model
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
    
    def train(self, train_loader, test_loader, criterion, optimizer, epochs):
        """
        Train the model
        
        Returns:
            History dictionary with training metrics
        """
        self.model.train()
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training phase
            epoch_train_loss = 0.0
            correct = 0
            total = 0
            
            for batch_X, batch_Y in train_loader:
                batch_X, batch_Y = batch_X.to(self.device), batch_Y.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_Y.float())
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Calculate metrics
                epoch_train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                _, labels = torch.max(batch_Y.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            # Calculate training accuracy and loss
            train_accuracy = 100 * correct / total
            avg_train_loss = epoch_train_loss / len(train_loader)
            
            # Evaluation phase
            test_accuracy, avg_test_loss = self.evaluate(test_loader, criterion)
            
            # Store history
            self.train_losses.append(avg_train_loss)
            self.train_accuracies.append(train_accuracy)
            self.test_losses.append(avg_test_loss)
            self.test_accuracies.append(test_accuracy)
            
            # Print progress
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, '
                  f'Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%')
        
        training_time = time.time() - start_time
        print(f'Training completed in {training_time:.2f} seconds')
        
        return {
            'train_loss': self.train_losses,
            'test_loss': self.test_losses,
            'train_accuracy': self.train_accuracies,
            'test_accuracy': self.test_accuracies,
            'training_time': training_time
        }
    
    def evaluate(self, test_loader, criterion):
        """Evaluate the model on test data"""
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_Y in test_loader:
                batch_X, batch_Y = batch_X.to(self.device), batch_Y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_Y.float())
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                _, labels = torch.max(batch_Y.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = test_loss / len(test_loader)
        
        return accuracy, avg_loss
    
    def plot_training_history(self, title, save_path=None):
        """Plot training history including loss and accuracy curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(title, fontsize=16)
        
        # Plot loss
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.test_losses, label='Test Loss')
        ax1.set_title('Loss Curves')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.test_accuracies, label='Test Accuracy')
        ax2.set_title('Accuracy Curves')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()