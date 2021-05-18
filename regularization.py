import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import  make_moons

np.random.seed(800) # to get the same result as mine
X, y = make_moons(n_samples=100, noise=0.2, random_state=1) # randomly initializing data
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33,
                                                    random_state=42)

reg_model = Sequential()
reg_model.add(Dense(500, input_dim=2, activation='relu',  kernel_regularizer='l2')) # l2 regularizer, experiment diff losses
reg_model.add(Dense(200, input_dim=2, activation='relu',  kernel_regularizer='l2'))
reg_model.add(Dense(100, input_dim=2, activation='relu',  kernel_regularizer='l2'))
reg_model.add(Dense(1, activation='sigmoid', kernel_regularizer='l2'))
reg_model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

reg_history = reg_model.fit(X_train, y_train,
                            validation_data=(X_test, y_test),
                            epochs=500, verbose=1)
print(reg_model.evaluate(X_test,y_test))
