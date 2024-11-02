import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from perceptron import Perceptron
from adeline import Adeline
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from testing import *


# Load your dataset
df = pd.read_csv(r"D:\UNI\NN\Task 1\NN-Tasks\preprocessed_birds.csv")
print(df)
class_mapping = {
    "A": 0,
    "B": 1,
    "C": 2
}

root = tk.Tk()
root.title("Bird Species Classification")

def plot_confusion_matrix(cm):
    # Create a new window for the confusion matrix
    cm_window = tk.Toplevel(root)
    cm_window.title("Confusion Matrix")

    fig, ax = plt.subplots(figsize=(10, 6))  # Increased figure size
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, f'{value}', ha='center', va='center')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()  # Ensure everything fits without overlap
    
    # Plot on a canvas in the new window
    canvas = FigureCanvasTkAgg(fig, master=cm_window)
    canvas.draw()
    canvas.get_tk_widget().pack()

def plot_decision_boundary(X, y, model):
    # Create a new window for the decision boundary
    boundary_window = tk.Toplevel(root)
    boundary_window.title("Decision Boundary")

    fig, ax = plt.subplots(figsize=(15, 15))  # Increased figure size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x_values = np.linspace(x_min, x_max, 100)

    # Ensure we are handling the correct number of features
    if X.shape[1] == 2:
        # Calculate decision boundary only if there are two features
        y_values = -(model.w_[0] * x_values + model.bias_) / model.w_[1]  # Equation of the line

        # Plot the points for the two classes
        ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', marker='o', label='Class 1')
        ax.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', marker='x', label='Class -1')

        # Plot decision boundary
        ax.plot(x_values, y_values, color='green', linestyle='--', label='Decision Boundary')
    
    else:
        # For higher dimensions, you may want to handle them differently or provide a warning
        ax.text(0.5, 0.5, "Decision boundary visualization only supported for 2D data.", 
                horizontalalignment='center', verticalalignment='center', fontsize=12, color='black')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend(loc='best')
    ax.set_title("Decision Boundary for Perceptron Model")
    ax.grid()

    # Plot on a canvas in the new window
    canvas = FigureCanvasTkAgg(fig, master=boundary_window)
    canvas.draw()
    canvas.get_tk_widget().pack()

def train(selected_features, selected_classes, learning_rate, epochs, bias, algorithm,mse_threshold):
    print("Original DataFrame shape:", df.shape)
    print("Unique values in 'bird category':", df['bird category'].unique())
    
    # Parse selected classes
    class_labels = selected_classes.split(" & ")

    # Map class labels to indices
    class_indices = [class_mapping[label.strip()] for label in class_labels]
    
    # Filter DataFrame based on selected classes
    df_filtered = df[df['bird category'].isin(class_indices)]
    print("Filtered DataFrame shape:", df_filtered.shape)

    print("DataFrame Columns:", df_filtered.columns)
    print("Selected Features:", selected_features)

    # Extract features and labels
    X = df_filtered[selected_features].to_numpy()
    y = df_filtered['bird category'].to_numpy()
    
    # Verify feature and label shapes
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print(X)
    print(y)

    # Convert classes to -1 and 1
    y = np.where(y == class_indices[0], -1, 1) 

    # Call custom_train_test_split with defined sizes
    (X_train, y_train), (X_test, y_test) = TTS(X, y)
    
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    # Initialize and fit the perceptron model
    if algorithm == 'Perceptron':
        model = Perceptron(eta=learning_rate, n_iter=epochs,init_bias=bias)
    else:
        model = Adeline(eta=learning_rate, n_iter=epochs,init_bias=bias,init_threshold=mse_threshold)

    model.fit(X_train, y_train)
    print(model)
    # Make predictions
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate accuracy
    accuracy = calculate_accuracy(y_test, y_pred)
    print("Predictions:", y_pred)
    print("True labels:", y_test)
    print("Calculated Accuracy:", accuracy)

    # Update accuracy label
    messagebox.showinfo("Training Result", f"Accuracy: {accuracy:.2f}%")

    # Plot confusion matrix
    plot_confusion_matrix(cm)
        
    # Plot decision boundary
    plot_decision_boundary(X_test, y_test, model)
        

def submit():
    selected_features = [feature for feature, var in feature_vars.items() if var.get()]
    
    if len(selected_features) != 2:
        messagebox.showwarning("Warning", "Please select exactly two features.")
        return
    
    selected_classes = class_var.get()  # Get the selected classes as a string
    learning_rate = learning_rate_var.get()
    epochs = epochs_var.get()
    mse_threshold = mse_threshold_var.get()
    bias = bias_var.get()
    algorithm=algorithm_var.get()
    # Display the entered values
    message = (f"Selected Features: {', '.join(selected_features)}\n"
               f"Selected Classes: {selected_classes}\n"
               f"Learning Rate: {learning_rate}\n"
               f"Number of Epochs: {epochs}\n"
               f"MSE Threshold: {mse_threshold}\n"
               f"Add Bias: {bias}")

    messagebox.showinfo("Submitted Data", message)

    # Train the Perceptron model
    train(selected_features, selected_classes, learning_rate, epochs, bias, algorithm,mse_threshold)

# Feature Selection
feature_label = tk.Label(root, text="Select Two Features:")
feature_label.pack()

features = ["gender", "body_mass", "beak_length", "beak_depth", "fin_length"]
feature_vars = {feature: tk.BooleanVar() for feature in features}

for feature in features:
    cb = tk.Checkbutton(root, text=feature, variable=feature_vars[feature])
    cb.pack(anchor=tk.W)

# Class Selection
class_label = tk.Label(root, text="Select Two Classes (C1 & C2 or C1 & C3 or C2 & C3):")
class_label.pack()

class_var = tk.StringVar(value="A & B")  # Default value
classes = ["A & B", "A & C", "B & C"]
for class_option in classes:
    rb = tk.Radiobutton(root, text=class_option, variable=class_var, value=class_option)
    rb.pack(anchor=tk.W)

# Learning Rate
learning_rate_label = tk.Label(root, text="Enter Learning Rate (eta):")
learning_rate_label.pack()
learning_rate_var = tk.DoubleVar(value=0.01)  # Default value
learning_rate_entry = tk.Entry(root, textvariable=learning_rate_var)
learning_rate_entry.pack()

# Number of Epochs
epochs_label = tk.Label(root, text="Enter Number of Epochs (m):")
epochs_label.pack()
epochs_var = tk.IntVar(value=100)  # Default value
epochs_entry = tk.Entry(root, textvariable=epochs_var)
epochs_entry.pack()

# MSE Threshold
mse_threshold_label = tk.Label(root, text="Enter MSE Threshold:")
mse_threshold_label.pack()
mse_threshold_var = tk.DoubleVar(value=0.01)  # Default value
mse_threshold_entry = tk.Entry(root, textvariable=mse_threshold_var)
mse_threshold_entry.pack()


# Add Bias
bias_var = tk.BooleanVar(value=False)  # Default value
bias_checkbox = tk.Checkbutton(root, text="Add Bias", variable=bias_var)
bias_checkbox.pack()

# Algorithm Selection
algorithm_label = tk.Label(root, text="Choose Algorithm:")
algorithm_label.pack()

algorithm_var = tk.StringVar(value="Perceptron")  # Default value
algorithms = ["Perceptron", "Adaline"]
for algorithm in algorithms:
    rb = tk.Radiobutton(root, text=algorithm, variable=algorithm_var, value=algorithm)
    rb.pack(anchor=tk.W)

# Submit Button
submit_button = tk.Button(root, text="Submit", command=submit)
submit_button.pack()

# Start the GUI event loop
root.mainloop()