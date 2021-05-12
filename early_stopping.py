import tensorflow as tf
import os
import time
import pandas as pd
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

df = pd.read_csv('data.csv')
df['Gender'].replace(['Female','Male'],[0,1], inplace=True)

target = "Purchased"
X = df.loc[:, df.columns != target]
y = df.loc[:, df.columns == target]
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=7)
scaler = StandardScaler().fit(X_train)
X_train_Scaled = scaler.transform(X_train)
X_test_Scaled = scaler.transform(X_test)

model = Sequential()
model.add(Dense(500, input_dim=3, activation='relu',  kernel_regularizer='l2'))
model.add(Dense(400,  activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])



model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
early_stopping_cb = EarlyStopping(patience=100,
                                  restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=1000,
                    validation_data=(X_test, y_test),
                    callbacks=early_stopping_cb)