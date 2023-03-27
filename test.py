from tkinter import Y
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./Salary_Data.csv')
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

print(x)
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print('LR init')

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print(y_pred)
print(y_test)

print('Accuracy: ', lr.score(x_train, y_train))

plt.scatter(x_train, y_train)
plt.plot(x_test, y_pred, color='red')

from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
regr = make_pipeline(StandardardScaler(), SVR(C=1.0, epsilon=0.2))
regr.fit(x_train, y_train)
y_pred1 = regr.predict(x_test)
print('Accuracy: ', lr.score(x_test, y_test))
plt.scatter(x_test, y_test)
plt.plot(x_test, y_pred1, color='red')
plt.show()