"""
utils.py

utility functions for training and testing
"""

import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def accuracy(outputs, labels):
    total = 0
    correct = 0
    _, predicted = torch.max(outputs.data, 1) ##
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    return correct/total

