# Bird Species Classification using Perceptron & Adaline

## Project Overview
This project implements a **GUI-based machine learning classifier** to distinguish between two bird species using either the **Perceptron** or **Adaline** learning algorithm. Users can select two features and two classes for classification, define training parameters, and visualize the decision boundary after training.

## Features
- **User-configurable inputs:**
  - Select **2 features** from: `gender`, `body_mass`, `beak_length`, `beak_depth`, `fin_length`.
  - Select **2 bird species** (C1 & C2, C1 & C3, or C2 & C3).
  - Set **learning rate (eta)**.
  - Define **number of epochs (m)**.
  - Set **MSE threshold** for Adaline.
  - Choose to **include bias** (checkbox).
  - Select algorithm: **Perceptron or Adaline** (radio button).

- **Training & Testing Process:**
  - Each class consists of **50 samples**.
  - **Training set:** 30 randomly selected samples per class.
  - **Testing set:** Remaining 20 samples per class.
  - **Weight initialization:** Small random values.
  
- **Classification & Evaluation:**
  - Train using **Perceptron or Adaline**.
  - Plot **decision boundary** and scatter points of both classes.
  - Compute **confusion matrix** and **overall accuracy**.
  - Classify a **single test sample** via GUI input.
