import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import pandas as pd

import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from matplotlib import pyplot as plt

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

train = pd.read_csv('carbon_nanotubes.csv')
print('Shape of the train data with all features:', train.shape)
train.replace(',','.', inplace = True, regex=True)
print("")
print('Shape of the train data with numerical features:', train.shape)

train = train.apply(pd.to_numeric)

X_set = train[['Chiral indice n','Chiral indice m','Initial atomic coordinate u', 'Initial atomic coordinate v', 'Initial atomic coordinate w']]
y_set = train[["Calculated atomic coordinates u'", "Calculated atomic coordinates v'", "Calculated atomic coordinates w'"]]

X_train, X_test, y_train, y_test = train_test_split(X_set, y_set, test_size=0.2)

print(X_set.dtypes)
print(y_set.dtypes)

X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

model = Sequential()
model.add(Dense(20, input_shape=(5,), activation = 'softmax'))
model.add(Dense(3,))
model.compile(Adam(lr=0.02), 'mean_squared_error')
history = model.fit(X_train, y_train, epochs = 1000, validation_split = 0.2,verbose = 0)

history_dict=history.history
loss_values = history_dict['loss']
val_loss_values=history_dict['val_loss']
plt.plot(loss_values,'bo',label='training loss')
plt.plot(val_loss_values,'r',label='training loss val')
plt.show()

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred)))
print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_test_pred)))