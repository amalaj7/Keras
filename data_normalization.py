import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

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
model.add(Dense(150, input_dim=3, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['binary_accuracy'])
#print(mode.summary())
history = model.fit(X_train_Scaled, y_train, epochs = 100,
                    validation_data = (X_test_Scaled, y_test))
