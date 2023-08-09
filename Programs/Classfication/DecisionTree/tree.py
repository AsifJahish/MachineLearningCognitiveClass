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


my_data= pd.read_csv("/home/asifjahish/MachineLearning/venv/MachineLearningCognitiveClass/Data/drug200.csv")
# print(my_data.head())


# what is the size of the Data and is (200 ,6) which means that 6 column and 200 rows
print(my_data.shape)


X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
print(X[0:5])

# As you may figure out, some features in this dataset are categorical, such as Sex or BP.
# Unfortunately, Sklearn Decision Trees does not handle categorical variables. We can still convert these features to
# numerical values using pandas.get_dummies() to convert the categorical variable into dummy/indicator variables.

le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1])


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3])

print(X[0:5])
# Now we can fill the target variable.
y = my_data["Drug"]
print(y[0:5])


#
# Now train_test_split will return 4 different parameters. We will name them:
# X_trainset, X_testset, y_trainset, y_testset
#
# The train_test_split will need the parameters:
# X, y, test_size=0.3, and random_state=3.
#
# The X and y are the arrays required before the
# split, the test_size represents the ratio of the testing dataset, and the random_state
# ensures that we obtain the same splits.


X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

print('Shape of X training set {}'.format(X_trainset.shape),'&',' Size of Y training set {}'.format(y_trainset.shape))


print('Shape of X training set {}'.format(X_testset.shape),'&',' Size of Y training set {}'.format(y_testset.shape))