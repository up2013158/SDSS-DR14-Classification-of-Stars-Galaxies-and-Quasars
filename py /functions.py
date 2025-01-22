import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def test_size_decision_tree(url, features, target):
    """
    Load data from a URL, train a decision tree classifier on varying test sizes,
    and plot the results.

    Parameters:
    - url: The URL of the dataset.
    - features: List of feature column names.
    - target: Target column name.
    """
    # Load the dataset
    data = pd.read_csv(url)

    # Select features and target variable
    X = data[features]
    Y = data[target]

    # Initialize lists to store results
    test_sizes = np.arange(0.01, 1.0, 0.01)  # Test sizes from 1% to 99%
    train_accuracies = []
    test_accuracies = []

    # Loop over test sizes
    for test_size in test_sizes:
        # Split the data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
        
        # Train a decision tree classifier
        dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
        dt_model.fit(X_train, Y_train)
        
        # Calculate training accuracy
        Y_train_pred = dt_model.predict(X_train)
        train_acc = accuracy_score(Y_train, Y_train_pred)
        train_accuracies.append(train_acc)
        
        # Calculate testing accuracy
        Y_test_pred = dt_model.predict(X_test)
        test_acc = accuracy_score(Y_test, Y_test_pred)
        test_accuracies.append(test_acc)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(test_sizes, train_accuracies, label='Training Accuracy', marker='o', color='blue')
    plt.plot(test_sizes, test_accuracies, label='Testing Accuracy', marker='o', color='orange')
    plt.xlabel('Test Size')
    plt.ylabel('Accuracy')
    plt.title('Dependence of Accuracy on Training and Test Sample Sizes')
    plt.grid(True)
    plt.legend()
    plt.show()

