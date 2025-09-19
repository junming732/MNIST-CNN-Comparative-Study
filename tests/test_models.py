

import torch
import pytest
from models.mlp import MLP
from models.cnn import SimpleCNN

def test_mlp_initialization():
    """Test that MLP initializes correctly"""
    model = MLP()
    assert model is not None
    assert sum(p.numel() for p in model.parameters()) > 0

def test_cnn_initialization():
    """Test that CNN initializes correctly"""
    model = SimpleCNN()
    assert model is not None
    assert sum(p.numel() for p in model.parameters()) > 0

def test_mlp_forward_pass():
    """Test MLP forward pass with sample input"""
    model = MLP()
    test_input = torch.randn(1, 784)
    output = model(test_input)
    assert output.shape == (1, 10)

def test_cnn_forward_pass():
    """Test CNN forward pass with sample input"""
    model = SimpleCNN()
    test_input = torch.randn(1, 784)
    output = model(test_input)
    assert output.shape == (1, 10)

def test_cnn_forward_pass_with_image():
    """Test CNN forward pass with image input"""
    model = SimpleCNN()
    test_input = torch.randn(1, 1, 28, 28)  # Batch, Channels, Height, Width
    output = model(test_input)
    assert output.shape == (1, 10)