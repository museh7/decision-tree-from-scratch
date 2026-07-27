# Decision Tree From Scratch
Decision Tree From Scratch

This project implements a Decision Tree classifier entirely from scratch in Python without using scikit-learn. The goal is to understand how decision trees work internally by building every stage of the algorithm manually.

Objectives
Understand how decision trees classify data.
Implement the Gini impurity calculation.
Find the best feature and threshold to split a dataset.
Build a decision tree recursively.
Predict the class of unseen data.
Compare the implementation with scikit-learn.
How a Decision Tree Works

A decision tree repeatedly asks simple questions about the data.

For example:

subscription_length_months < 8?

If the answer is Yes, the customer moves to the left branch.

If the answer is No, the customer moves to the right branch.

The algorithm searches through many possible questions and chooses the one that produces the purest split of the target labels. Purity is measured using Gini impurity.

The same process is then repeated on each branch until the data is sufficiently separated or a stopping condition is reached.
