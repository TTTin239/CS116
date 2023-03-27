import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Advanced Performance Evluation Method
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score

# Get the 10 accuracies for each one of the 10 combinations that will be created through k-fold cross validation
accuracies = cross_val_score(estimator = classifier, 
                             X = X_train, 
                             y= y_train, #dependent variable vector of the training set 
                             cv = 10)   # number of folds you want to split your training set into. most common choice is 10 because 10 accuracies is enough to get a relevant idea of model performance

# Get average of the 10 accuracies of the accuracies vector
accuracies.mean() # relevant evaluation of model performance

# Standard deviation of accuracies variance
accuracies.std() 

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV

# Build parameters list of dictionaries that contains different options to be investigated by grid search to find the best set of parameters
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']}, # investigate linear model option
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.5, 0.1, 0.01, 0.001, 0.0001]} #non-linear model option and investigate several sub-options of penalty parameters and gamma
              ]

# Create grid search object
grid_search = GridSearchCV(estimator = classifier, # machine learning model
                           param_grid = parameters, 
                           scoring = 'accuracy', # scoring metric we're going to use to decide what the best parameters are (could be accuracy, precision, recall)
                           cv = 10, # so 10 fold cross validation will be applied through grid search
                           n_jobs = -1) # for large data sets

# Fit grid search object to training set
grid_search = grid_search.fit(X_train, y_train)

# View grid search results
best_accuracy = grid_search.best_score_ # mean of 10 accuracies measured through 10-fold cross-validation
best_parameters = grid_search.best_params_ # can tweak sub-options in paramaters based on this result to improve model performance even more

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()