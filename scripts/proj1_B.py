############################################################################################
# The following program deals with the problem 2 of project 1. The program implements the  #
# machine learning algorithms: perceptron, logistic regression, support vector machines,   #
# decision trees, random forests and k-nearest neighbor.                                   #
# With the model trained on subsequent algorithms, the accuracy of each was compared.      #
############################################################################################

import numpy as np  
import pandas as pd                                   
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split


# function to determine the data metrics to be displayed
def analysis(y_test, y_pred, y_combined, y_combined_pred):
   
    print('Incorrectly classified training samples: %d' % (y_test != y_pred).sum())
    print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))
    print('Combined (train & test) incorrectly classified samples: %d' % (y_combined != y_combined_pred).sum())
    print('Combined (train & test) Accuracy: %.4f' % accuracy_score(y_combined, y_combined_pred))


# function to standardize testing and training data
def standardize(xtrain, xtest):
    
    scalar = StandardScaler()                                                    # defining a standard scalar
    scalar.fit(xtrain)
    std_xtrain = scalar.transform(xtrain)                                        # standardizing training data
    std_xtest = scalar.transform(xtest)                                          # standardizing testing data
    std_xComb = np.vstack((std_xtrain, std_xtest))
    xComb = np.vstack((xtrain, xtest))

    return std_xtrain, std_xtest, std_xComb, xComb


# function to implement perceptron algorithm
def perceptron(xtrain, xtest, ytrain, ytest):
    
    std_xtrain, std_xtest, std_xComb, xComb = standardize(xtrain, xtest)
    perceptron = Perceptron(max_iter=10, tol=1e-2, eta0=1e-3, fit_intercept=True, random_state=0, verbose=False)
    perceptron.fit(std_xtrain, ytrain.values.ravel())                            # training step
    ypredict = perceptron.predict(std_xtest)                                     # prediction step
    predict_yComb = perceptron.predict(std_xComb)

    return ypredict, predict_yComb


# function to implement logistic regression algorithm
def logistic_regression(xtrain,xtest,ytrain,ytest):

    std_xtrain, std_xtest, std_xComb, xComb = standardize(xtrain, xtest)
    loreg = LogisticRegression(C=10.0, solver='lbfgs', multi_class='ovr', random_state=1)
    loreg.fit(std_xtrain, ytrain)                                                   # training step
    ypredict = loreg.predict(std_xtest)                                             # prediction step
    predict_yComb = loreg.predict(std_xComb)
    
    return ypredict, predict_yComb


# function to implement support vector machines algorithm
def supportvectormachine(xtrain,xtest,ytrain,ytest):
    
    std_xtrain, std_xtest, std_xComb, xComb = standardize(xtrain, xtest)
    svm = SVC(kernel='linear', C=3.0, random_state=0)
    svm.fit(std_xtrain, ytrain)                                                     # training step
    ypredict = svm.predict(std_xtest)                                               # prediction step
    predict_yComb = svm.predict(std_xComb)

    return ypredict, predict_yComb


# function to implement decision tree algorithm
def decisiontree(xtrain,xtest,ytrain,ytest):
    
    std_xtrain, std_xtest, std_xComb, xComb = standardize(xtrain, xtest)
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0)
    tree.fit(xtrain, ytrain)                                                        # training step
    ypredict = tree.predict(xtest)                                                  # prediction step
    predict_yComb = tree.predict(xComb)
   
    return ypredict, predict_yComb


# function to implement random forests algorithm
def randomforest(xtrain, xtest, ytrain, ytest):
    
    std_xtrain, std_xtest, std_xComb, xComb = standardize(xtrain, xtest)
    frst = RandomForestClassifier(criterion='entropy', n_estimators=10,random_state=1,min_samples_split = 5, n_jobs=2)
    frst.fit(xtrain, ytrain)                                                       # training step
    ypredict = frst.predict(xtest)                                                 # prediction step
    predict_yComb = frst.predict(xComb)
   
    return ypredict, predict_yComb


# function to implement k nearest neighbor algorithm
def knearestneighbor(xtrain, xtest, ytrain, ytest):

    std_xtrain, std_xtest, std_xComb, xComb = standardize(xtrain, xtest)
    knn = KNeighborsClassifier(n_neighbors=20,algorithm='auto', p=2, metric='minkowski')
    knn.fit(std_xtrain, ytrain)                                                    # training step
    ypredict = knn.predict(std_xtest)                                              # prediction step
    predict_yComb = knn.predict(std_xComb)
    
    return ypredict, predict_yComb


def algorithms(algo, xtrain, xtest, ytrain, ytest):

    if algo == 'perceptron':
        print('PERCEPTRON')
        ypredict, predict_yComb = perceptron(xtrain, xtest, ytrain, ytest)

    elif algo == 'logisticregression':
        print('LOGISTIC REGRESSION')
        ypredict, predict_yComb = logistic_regression(xtrain, xtest, ytrain, ytest)

    elif algo == 'supportvectormachine':
        print('SUPPORT VECTOR MACHINES')
        ypredict, predict_yComb = supportvectormachine(xtrain, xtest, ytrain, ytest)

    elif algo == 'decisiontrees':
        print('DECISION TREES')
        ypredict, predict_yComb  = decisiontree(xtrain, xtest, ytrain, ytest)

    elif algo == 'randomforests':
        print('RANDOM FORESTS')
        ypredict, predict_yComb  = randomforest(xtrain, xtest, ytrain, ytest)

    elif algo == 'knearestneighbor':
        print('K NEAREST NEIGHBORS')
        ypredict, predict_yComb  = knearestneighbor(xtrain, xtest, ytrain, ytest)

    else:
        print('Out of Scope')
        
    return ypredict, predict_yComb


file = '..\\resources\\data_banknote_authentication.txt'
dat = pd.read_csv(file)
df = pd.DataFrame(dat)
X = df.iloc[:, 0:3]                          # considering all features
Y = df.iloc[:, 4]


xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.3, random_state=0)      #splitting the data into train and test
methods = ['perceptron','logisticregression','supportvectormachine','decisiontrees','randomforests','knearestneighbor']

for method in methods:
    ypredict, predict_yComb = algorithms(method,xtrain, xtest, ytrain, ytest)
    yComb = np.hstack((ytrain, ytest))
    analysis(ytest, ypredict, yComb, predict_yComb)
    print()
