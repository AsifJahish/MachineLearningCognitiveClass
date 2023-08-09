def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import sys
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn import metrics
import matplotlib.pyplot as plt


my_data= pd.read_csv("/home/asifjahish/MachineLearning/venv/MachineLearningCognitiveClass/Data/drug200.csv")



# print(my_data.shape)


X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values


le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1])


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3])

# print(X[0:5])
# Now we can fill the target variable.
y = my_data["Drug"]
# print(y[0:5])




X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

print('Shape of X training set {}'.format(X_trainset.shape),'&',' Size of Y training set {}'.format(y_trainset.shape))


# Modeling
# We will first create an instance of the DecisionTreeClassifier called drugTree.
# Inside of the classifier, specify criterion="entropy" so we can see the information gain of each node.


drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
print(drugTree)

print(drugTree.fit(X_trainset,y_trainset))


predTree = drugTree.predict(X_testset)

print (predTree [0:5])
print (y_testset [0:5])

print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))


tree.plot_tree(drugTree)
plt.title("the Tree is Amazing to see like this ")
plt.show()