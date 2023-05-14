# KNN Classifier with Iris Dataset

This Python code demonstrates how to use the KNN algorithm to classify iris flowers into three different species based on the length and width of their petals and sepals.

The program first loads the iris dataset from the UCI Machine Learning Repository and splits it into training and testing sets. It then creates a KNN classifier with k=5 and uses k-fold cross-validation to evaluate its performance.

After training the model on the entire training set, the program makes predictions on the testing set and generates a confusion matrix and classification report to evaluate the model's performance. The confusion matrix is also visualized as a heatmap using Matplotlib.

## Requirements

- Python 3
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## Usage

1. Install the required dependencies using pip:

2. Download the iris dataset from the UCI Machine Learning Repository:

3. Save the `iris.data` file in the same directory as the Python script.

4. Run the Python script:

The program will output the cross-validation scores, mean score, confusion matrix, and classification report, as well as save a visualization of the confusion matrix as a PNG file named `knn.png`.
"""

with open("README.md", "w") as f:
 f.write(readme)
