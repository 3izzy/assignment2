import numpy as np
import pandas as pd
import patsy as pt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")
test_data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv").drop(['id', 'DateTime', 'meal'], axis=1)
Y = data['meal']
X = data.drop(['id', 'DateTime', 'meal'], axis=1)

model = tree.DecisionTreeClassifier()
modelFit = model.fit(X, Y)
pred = modelFit.predict(test_data)