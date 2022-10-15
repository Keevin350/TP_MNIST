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


# In[3]:


mnist.data


# In[4]:


print(mnist.target.shape)
print(mnist.data.shape)


# In[5]:


import matplotlib.pyplot as plt

lotImage = mnist.data.to_numpy()

for i in range(10):
    plt.imshow((lotImage[i].reshape(28,28)), cmap=plt.cm.gray_r)
    plt.show()
    


# In[6]:


import time 
from sklearn.metrics import accuracy_score

#faire un PCA REDUCTION DE DIMENSION 

X_train = mnist.data[:60000]
X_test = mnist.data[60000:]
y_train = mnist.target[:60000]
y_test = mnist.target[60000:]

print("X_train : ", len(X_train), " y_train : ", len(y_train))
print("X_test : ", len(X_test), " y_test : ", len(y_test))


# In[52]:


start_time = time.time()
mlp = MLPClassifier()
mlp.fit(X_train, y_train)
prediction = mlp.predict(X_test)
print(accuracy_score(y_test, prediction))
print("--- %s seconds ---" % (time.time() - start_time))


# In[7]:


scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.fit_transform(X_test)

start_time = time.time()
mlp = MLPClassifier()
mlp.fit(X_train_scaler, y_train)
prediction = mlp.predict(X_test_scaler)
print(accuracy_score(y_test, prediction))
print("--- %s seconds ---" % (time.time() - start_time))


# In[8]:


from sklearn.decomposition import PCA
import numpy as np

pca = PCA(n_components=784)
X_train_reduction_scaler = pca.fit_transform(X_train_scaler)
plt.plot(np.cumsum(pca.explained_variance_ratio_))


# In[9]:


pca = PCA(n_components=500)
X_train_reduction_scaler = pca.fit_transform(X_train_scaler)

parametres = {'hidden_layer_sizes':[(50,),(50,50),(50,50,50)],
             'alpha':[0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1],
             'learning_rate':["constant","invscaling","adaptive"]}

mlp_clf = MLPClassifier()
gridS_mlp = GridSearchCV(mlp_clf, parametres, cv=2, n_jobs=-1)
gridS_mlp.fit(X_train_reduction_scaler, y_train)


# In[10]:


print(gridS_mlp.best_params_)


# In[23]:


start_time = time.time()
mlp = MLPClassifier(hidden_layer_sizes = (50,), alpha =0.1 ,learning_rate = "invscaling")
mlp.fit(X_train_scaler, y_train)
prediction = mlp.predict(X_test_scaler)
print(accuracy_score(y_test, prediction))
print("--- %s seconds ---" % (time.time() - start_time))


# In[25]:


MaximumScore = 0

for i in range(0,10):
    print(i)
    start_time = time.time()
    mlp = MLPClassifier(hidden_layer_sizes = (100,), alpha =0.1 ,learning_rate = "invscaling")
    mlp.fit(X_train_scaler, y_train)
    prediction = mlp.predict(X_test_scaler)
    
    if MaximumScore < accuracy_score(y_test, prediction):
        MaximumScore = accuracy_score(y_test, prediction)
        print(accuracy_score(y_test, prediction))
        print("--- %s seconds ---" % (time.time() - start_time))


# In[26]:


MaximumScore = 0

for i in range(0,5):
    print(i)
    start_time = time.time()
    mlp = MLPClassifier(hidden_layer_sizes = (500,), alpha =0.1 ,learning_rate = "invscaling")
    mlp.fit(X_train_scaler, y_train)
    prediction = mlp.predict(X_test_scaler)
    
    if MaximumScore < accuracy_score(y_test, prediction):
        MaximumScore = accuracy_score(y_test, prediction)
        print(accuracy_score(y_test, prediction))
        print("--- %s seconds ---" % (time.time() - start_time))


# In[ ]:




