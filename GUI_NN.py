import tkinter as tk
from tkinter import messagebox

# Function to handle submission
def submit():
    selected_features = [feature for feature, var in feature_vars.items() if var.get()]
    if len(selected_features) != 2:
        messagebox.showwarning("Warning", "Please select exactly two features.")
        return
    
    classes = class_var.get()
    learning_rate = learning_rate_var.get()
    epochs = epochs_var.get()
    mse_threshold = mse_threshold_var.get()
    bias = bias_var.get()
    algorithm = algorithm_var.get()

    # Display the entered values
    message = (f"Selected Features: {', '.join(selected_features)}\n"
               f"Selected Classes: {classes}\n"
               f"Learning Rate: {learning_rate}\n"
               f"Number of Epochs: {epochs}\n"
               f"MSE Threshold: {mse_threshold}\n"
               f"Add Bias: {bias}\n"
               f"Algorithm: {algorithm}")

    messagebox.showinfo("Submitted Data", message)

# Create the main window
root = tk.Tk()
root.title("Bird Species Classification")

# Feature Selection
feature_label = tk.Label(root, text="Select Two Features:")
feature_label.pack()

features = ["Gender", "Body Mass", "Beak Length", "Beak Depth", "Fin Length"]
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
