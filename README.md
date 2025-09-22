# MNIST Classification: A Comparative Deep Learning Study

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)

A comprehensive implementation and analysis of various neural network architectures on the MNIST dataset.

## 📖 Overview

This project implements and benchmarks multiple models to classify handwritten digits from the MNIST dataset.

## 🚀 Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
 ```

2. Run an experiment
 ```bash
python train.py --model cnn --optimizer adam --epochs 10
 ```

 ## 📁 Code Structure

The project is modularized for clarity and reusability:
- `models/`: Contains PyTorch module definitions for all architectures.
- `utils/`: Contains boilerplate code for training, data loading, and visualization.
- `main.py`: The main script to orchestrate experiments and parse command-line arguments.

*This project was completed as part of the Deep Learning (1RT720) course at Uppsala University.*
