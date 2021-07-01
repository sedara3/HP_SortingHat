#!/usr/bin/env python
# coding: utf-8

# # Hogwart's Sorting Hat Algorithm

# The aim of this project was to build a machine-learning-powered Harry Potter's Sorting Hat that could tell which Hogwarts House you belong to based on given features. In this notebook I've implemented several multi-class classification algorithms in Python. 

# In[173]:


pip install seaborn


# In[174]:


pip install statsmodels


# In[175]:


import pandas as pd
import numpy as np

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# machine learning
from sklearn.preprocessing import StandardScaler

import sklearn.linear_model as skl_lm
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn.model_selection import train_test_split


import statsmodels.api as sm
import statsmodels.formula.api as smf


# initialize some package settings
sns.set(style="whitegrid", color_codes=True, font_scale=1.3)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[176]:


df = pd.read_csv('dataset.csv') #importing dataset


# In[177]:


df


# In[178]:


df.info()


# In[179]:


#filling up the missing values using the averages of the columns
df['Arithmancy'].fillna((df['Arithmancy'].mean()), inplace=True)
df['Astronomy'].fillna((df['Astronomy'].mean()), inplace=True)
df['Herbology'].fillna((df['Herbology'].mean()), inplace=True)
df['Defense Against the Dark Arts'].fillna((df['Defense Against the Dark Arts'].mean()), inplace=True)
df['Divination'].fillna((df['Divination'].mean()), inplace=True)
df['Muggle Studies'].fillna((df['Muggle Studies'].mean()), inplace=True)
df['Ancient Runes'].fillna((df['Ancient Runes'].mean()), inplace=True)
df['History of Magic'].fillna((df['History of Magic'].mean()), inplace=True)
df['Transfiguration'].fillna((df['Transfiguration'].mean()), inplace=True)
df['Potions'].fillna((df['Potions'].mean()), inplace=True)
df['Care of Magical Creatures'].fillna((df['Care of Magical Creatures'].mean()), inplace=True)


# In[180]:


df.info()


# In[181]:


df.dtypes


# In[182]:


# visualize distribution of classes 
plt.figure(figsize=(8, 4))
sns.countplot(df['Hogwarts House'], palette='RdBu')

# count number of obvs in each class
Ravenclaw, Slytherin, Gryffindor, Hufflepuff = df['Hogwarts House'].value_counts()
print('Number of students in Ravenclaw: ', Ravenclaw)
print('Number of students in Slytherin: ', Slytherin)
print('Number of students in Gryffindor: ', Gryffindor)
print('Number of students in Hufflepuff: ', Hufflepuff)
print('')
print('% of cells labeled Ravenclaw', round(Ravenclaw / len(df) * 100, 2), '%')
print('% of cells labeled Slytherin', round(Slytherin / len(df) * 100, 2), '%')
print('% of cells labeled Gryffindor', round(Gryffindor / len(df) * 100, 2), '%')
print('% of cells labeled Hufflepuff', round(Hufflepuff / len(df) * 100, 2), '%')


# In[183]:


# generate a scatter plot matrix with the columns
cols = ['Hogwarts House',
    'Best Hand', 
        'Arithmancy', 
        'Astronomy', 
        'Herbology', 
        'Defense Against the Dark Arts', 
        'Divination', 
        'Muggle Studies',
        'Ancient Runes', 
        'History of Magic', 
        'Transfiguration',
       'Potions',
       'Care of Magical Creatures',
       'Charms',
       'Flying']

sns.pairplot(data=df[cols], hue='Hogwarts House', palette='RdBu')


# In[184]:


df = df.drop('Index', axis=1) #removing index column


# In[185]:


# Generate and visualize the correlation matrix
corr = df.corr().round(2)

# Mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set figure size
f, ax = plt.subplots(figsize=(20, 20))

# Define custom colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap
sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.tight_layout()


# In[186]:


#for label encoding
df['Hogwarts House'].unique()


# In[187]:


# Import label encoder
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df['Hogwarts House']= label_encoder.fit_transform(df['Hogwarts House']) 
df['Hogwarts House'].unique()


# Gryffindor-0 
# Hufflepuff-1
# Ravenclaw-2
# Slytherin-3 

# In[188]:


df['Best Hand'].unique()


# In[189]:


df['Best Hand']= label_encoder.fit_transform(df['Best Hand']) 
df['Best Hand'].unique()


# Left- 0
# Right- 1

# In[190]:


df = df.drop('First Name', axis=1) #removing unecessary data
df = df.drop('Last Name', axis=1)
df = df.drop('Birthday', axis=1)


# In[191]:


# Split the data into training and testing sets
X = df
y = df['Hogwarts House']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2)


# # Support vector machine classifier 

# In[192]:


# training a linear SVM classifier
from sklearn.svm import SVC
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train)
svm_predictions = svm_model_linear.predict(X_test)


# In[193]:


# model accuracy for X_test  
accuracy = svm_model_linear.score(X_test, y_test)
  
# creating a confusion matrix
cm = confusion_matrix(y_test, svm_predictions)


# In[194]:


print(accuracy)


# In[195]:


print(cm)


# In[196]:


from sklearn.metrics import plot_confusion_matrix
matrix = plot_confusion_matrix(svm_model_linear, X_test, y_test,cmap=plt.cm.Greys)
plt.title('Confusion matrix for SVM classifier')
plt.grid(False)
plt.show()


# # Decision tree classifier

# In[197]:


from sklearn.tree import DecisionTreeClassifier


# In[198]:


dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train)
dtree_predictions = dtree_model.predict(X_test)


# In[199]:


cm1 = confusion_matrix(y_test, dtree_predictions)
cm1


# In[200]:


accuracy1 = dtree_model.score(X_test, y_test)
accuracy1


# In[201]:


from sklearn.metrics import plot_confusion_matrix
matrix = plot_confusion_matrix(dtree_model, X_test, y_test,cmap=plt.cm.Greys)
plt.title('Confusion matrix for Decision Tree Classifier')
plt.grid(False)
plt.show()


# # KNN classifier

# In[202]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 4).fit(X_train, y_train)


# In[203]:


accuracy2 = knn.score(X_test, y_test)
print (accuracy2)


# In[204]:


knn_predictions = knn.predict(X_test) 
cm2 = confusion_matrix(y_test, knn_predictions)


# In[205]:


cm2


# In[206]:


matrix = plot_confusion_matrix(knn, X_test, y_test,cmap=plt.cm.Greys)
plt.title('Confusion matrix for KNN Classifier')
plt.grid(False)
plt.show()

