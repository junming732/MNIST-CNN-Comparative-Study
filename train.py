
"""
Main training script for MNIST classification experiments
"""

import argparse
import json
import torch
from utils.data_loader import MNISTDataLoader
from utils.trainer import CNNTrainer
from models.mlp import MLP
from models.cnn import SimpleCNN

def run_experiment(model_type='cnn', optimizer_name='sgd', 
                  learning_rate=0.01, epochs=10, batch_size=64, hidden_dim=280):
    """Run a complete experiment with the specified parameters"""
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    data_loader = MNISTDataLoader()
    X_train, Y_train, X_test, Y_test = data_loader.load_data()
    
    # Create datasets and dataloaders
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    if model_type.lower() == 'mlp':
        model = MLP(hidden_dim=hidden_dim)
    else:  # Default to CNN
        model = SimpleCNN()
    
    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    
    if optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:  # Default to SGD
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    # Train model
    trainer = CNNTrainer(model, device)
    history = trainer.train(train_loader, test_loader, criterion, optimizer, epochs)
    
    # Generate plots
    title = f"{model_type.upper()} with {optimizer_name.upper()} optimizer (LR: {learning_rate})"
    trainer.plot_training_history(title)
    
    # Calculate final metrics
    final_train_acc = history['train_accuracy'][-1]
    final_test_acc = history['test_accuracy'][-1]
    
    print(f"Final Training Accuracy: {final_train_acc:.2f}%")
    print(f"Final Test Accuracy: {final_test_acc:.2f}%")
    
    # Return results
    return {
        'model_type': model_type,
        'optimizer': optimizer_name,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'batch_size': batch_size,
        'final_train_accuracy': final_train_acc,
        'final_test_accuracy': final_test_acc,
        'training_time': history['training_time']
    }

def main():
    """Main function to run experiments based on command line arguments"""
    parser = argparse.ArgumentParser(description='MNIST Classification Experiments')
    parser.add_argument('--model', type=str, default='cnn', choices=['mlp', 'cnn'],
                        help='Model architecture to use')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'],
                        help='Optimizer to use')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--hidden_dim', type=int, default=280, 
                        help='Hidden dimension for MLP (ignored for CNN)')
    parser.add_argument('--output', type=str, default='results.json',
                        help='Output file to save results')
    
    args = parser.parse_args()
    
    # Run experiment
    results = run_experiment(
        model_type=args.model,
        optimizer_name=args.optimizer,
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim
    )
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()