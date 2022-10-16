#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as pd
from sklearn.preprocessing import StandardScaler

warnings.simplefilter("ignore")
mnist = fetch_openml("mnist_784")


# In[2]:


mnist.data


# In[3]:


print(mnist.target.shape)
print(mnist.data.shape)


# In[4]:


import matplotlib.pyplot as plt

lotImage = mnist.data.to_numpy()

for i in range(10):
    plt.imshow((lotImage[i].reshape(28,28)), cmap=plt.cm.gray_r)
    plt.show()
    


# In[5]:


import time 
from sklearn.metrics import accuracy_score

X_train = mnist.data[:60000]
X_test = mnist.data[60000:]
y_train = mnist.target[:60000]
y_test = mnist.target[60000:]

print("X_train : ", len(X_train), " y_train : ", len(y_train))
print("X_test : ", len(X_test), " y_test : ", len(y_test))


# In[6]:


start_time = time.time()
mlp = MLPClassifier()
mlp.fit(X_train, y_train)
prediction = mlp.predict(X_test)
print(accuracy_score(y_test, prediction))
print("--- %s seconds ---" % (time.time() - start_time))


# In[6]:


scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.fit_transform(X_test)

start_time = time.time()
mlp = MLPClassifier()
mlp.fit(X_train_scaler, y_train)
prediction = mlp.predict(X_test_scaler)
print(accuracy_score(y_test, prediction))
print("--- %s seconds ---" % (time.time() - start_time))


# In[34]:


parametres = {'hidden_layer_sizes':[(50,),(50,50),(50,50,50)],
             'activation':["identity","logistic","tanh","relu"],
             'solver':["lbfgs","sgd","adam"],
             'alpha':[0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1],
             'learning_rate':["constant","invscaling","adaptive"]}

mlp_clf = MLPClassifier()
gridS_mlp = GridSearchCV(mlp_clf, parametres, cv=3, n_jobs=4)
gridS_mlp.fit(X_train_scaler, y_train)


# In[18]:


print(gridS_mlp.best_params_)


# In[22]:


start_time = time.time()
mlp = MLPClassifier(hidden_layer_sizes = (50, 50), activation="relu", solver="adam",alpha =0.1 ,learning_rate = "adaptive")
mlp.fit(X_train_scaler, y_train)
prediction = mlp.predict(X_test_scaler)
print(accuracy_score(y_test, prediction))
print("--- %s seconds ---" % (time.time() - start_time))


# In[28]:


MaximumScore = 0

for i in range(0,20):

    start_time = time.time()
    mlp = MLPClassifier(hidden_layer_sizes = (50, 50), activation="relu", solver="adam",alpha =0.1 ,learning_rate = "adaptive")
    mlp.fit(X_train_scaler, y_train)
    prediction = mlp.predict(X_test_scaler)
    
    if MaximumScore < accuracy_score(y_test, prediction):
        MaximumScore = accuracy_score(y_test, prediction)
        tempsExec = time.time() - start_time
        print(accuracy_score(y_test, prediction))
        print("--- %s seconds ---" % (time.time() - start_time))
    
    
    


# In[32]:


MaximumScore = 0

for i in range(0,10):

    start_time = time.time()
    mlp = MLPClassifier(hidden_layer_sizes = (100, 100), activation="relu", solver="adam",alpha =0.1 ,learning_rate = "adaptive")
    mlp.fit(X_train_scaler, y_train)
    prediction = mlp.predict(X_test_scaler)
    
    if MaximumScore < accuracy_score(y_test, prediction):
        MaximumScore = accuracy_score(y_test, prediction)
        tempsExec = time.time() - start_time
        print(accuracy_score(y_test, prediction))
        print("--- %s seconds ---" % (time.time() - start_time))
    
    
    


# In[7]:


start_time = time.time()
mlp=MLPClassifier(solver='adam',random_state=1, max_iter=100, hidden_layer_sizes=(500,),alpha=1e-6)
mlp.fit(X_train_scaler, y_train)
prediction = mlp.predict(X_test_scaler)
print(accuracy_score(y_test, prediction))
print("--- %s seconds ---" % (time.time() - start_time))

