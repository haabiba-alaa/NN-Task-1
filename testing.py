import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def TTS(X, y, train_size_per_class=30, test_size_per_class=20):
    # Ensure y is a numpy array
    y = np.array(y)
    
    # Get indices for each class
    class_0_indices = np.where(y == -1)[0]
    class_1_indices = np.where(y == 1)[0]
    
    # Shuffle indices
    np.random.shuffle(class_0_indices)
    np.random.shuffle(class_1_indices)
    
    # Split for class -1 (class 0)
    train_indices_0 = class_0_indices[:train_size_per_class]
    test_indices_0 = class_0_indices[train_size_per_class:train_size_per_class + test_size_per_class]
    
    # Split for class 1
    train_indices_1 = class_1_indices[:train_size_per_class]
    test_indices_1 = class_1_indices[train_size_per_class:train_size_per_class + test_size_per_class]
    
    # Combine indices
    train_indices = np.concatenate([train_indices_0, train_indices_1])
    test_indices = np.concatenate([test_indices_0, test_indices_1])
    
    # Create training and testing datasets
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    return (X_train, y_train), (X_test, y_test)

def confusion_matrix(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("Length of y_true and y_pred must be the same.")
        
    # Initialize counts
    tp = tn = fp = fn = 0

    # Calculate counts based on predictions
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:  # True Positive
            tp += 1
        elif true == -1 and pred == -1:  # True Negative
            tn += 1
        elif true == -1 and pred == 1:  # False Positive
            fp += 1
        elif true == 1 and pred == -1:  # False Negative
            fn += 1

    return np.array([[tn, fp], [fn, tp]])

def calculate_accuracy(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("Length of y_true and y_pred must be the same.")

    correct_predictions = (y_true == y_pred).sum()
    accuracy = correct_predictions / len(y_true) * 100  # Convert to percentage
    return accuracy