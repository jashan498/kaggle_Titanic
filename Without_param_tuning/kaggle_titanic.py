# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 16:12:55 2017

@author: Jashan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## IMPORTING DATASET
dataset = pd.read_csv('train.csv')
X = np.array(dataset[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']])
y = dataset.iloc[:, 1].values



#NOW I HAVE TO TAKE CARE OF MISSING DATA TOO. 

from sklearn.preprocessing import Imputer 
imputer=Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer.fit(X[:,2:3])
X[:,2:3]=imputer.transform(X[:,2:3])

#ENCODING CATEGORICAL DATA. UFF THIS TAKES TIME.

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
labelencoder_X2=LabelEncoder()
X[:,1]=labelencoder_X.fit_transform(X[:,1])
X[:,6]=labelencoder_X2.fit_transform(X[:,6])

#LETS MAKE SOME DUMMY VARIABLE

onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()

onehotencoder2=OneHotEncoder(categorical_features=[1])
X=onehotencoder2.fit_transform(X).toarray()

onehotencoder3=OneHotEncoder(categorical_features=[6])
X=onehotencoder3.fit_transform(X).toarray()


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


##HERES THE FUN PART BEGINS...

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 16))
# classifier.add(Dropout(p = 0.1))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
# classifier.add(Dropout(p = 0.1))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

#ACCURACY
y_pred = classifier.predict(X_test)

for x in range(len(y_pred)):
    if(y_pred[x]<0.5):
        y_pred[x]=0
        
    else:
        y_pred[x]=1
        
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


####PREDICTING THE DATASET GIVEN BY KAGGLE

dataset_test = pd.read_csv('test.csv')

yo= np.array(dataset_test[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']])

dataset_test.isnull().any()

#dataset_test.dtypes




#NOW I HAVE TO TAKE CARE OF MISSING DATA TOO. 


from sklearn.preprocessing import Imputer 
imputer=Imputer(missing_values="NaN",strategy="mean",axis=0)

imputer.fit(yo[:,2:3])
yo[:,2:3]=imputer.transform(yo[:,2:3])

imputer.fit(yo[:,5:6])
yo[:,5:6]=imputer.transform(yo[:,5:6])



#ENCODING CATEGORICAL DATA. UFF THIS TAKES TIME.


yo[:,1]=labelencoder_X.fit_transform(yo[:,1])
yo[:,6]=labelencoder_X2.fit_transform(yo[:,6])

#LETS MAKE SOME DUMMY VARIABLE



yo=onehotencoder.fit_transform(yo).toarray()


yo=onehotencoder2.fit_transform(yo).toarray()


yo=onehotencoder3.fit_transform(yo).toarray()

#SCALING

yo = sc.transform(yo)


####TEST TIME !!

y_pred2 = classifier.predict(yo)

for x in range(len(y_pred2)):
    if(y_pred2[x]<0.5):
        y_pred2[x]=0
        
    else:
        y_pred2[x]=1
        
df = pd.DataFrame(y_pred2)
df.to_csv("rans.csv")

        
        
































