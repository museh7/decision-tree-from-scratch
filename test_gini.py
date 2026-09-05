from dectree import gini
import pandas as pd
print(gini([1, 1, 1, 1, 1]))
print(gini([7, 7, 7, 7]))
print(gini([0, 1]))
print(gini([0, 0, 1, 1]))
print(gini([0, 0, 0, 0, 1]))
print(gini([0, 1, 2]))
print(gini([0, 0, 0, 1, 2]))
print(gini(["cat", "dog", "dog", "cat"]))
print(gini(["A", "B", "C", "D"]))


