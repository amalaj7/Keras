#Importing Necessary Libraries

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

#Checking the version of Tensorflow & Keras
print(f"Tensorflow Version : {tf.__version__}")
print(f"Keras Version : {tf.keras.__version__}")

# to check the cpu and gpu
device_list = ['CPU', 'GPU']

for device in device_list:
  myDevice = tf.config.list_physical_devices(device)
  if len(myDevice) > 0:
    print(f"{device} is available!")
    print(f"Details: {myDevice}")

  else:
    print(f"{device} is unavailable!")
    print(f"Details: {myDevice}")


f_MNIST = tf.keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = f_MNIST.load_data()

# Creating a validatiion set from X_train_full and Scaling the data by 255. Cz, it's an uint8 data
X_valid, X_train = X_train_full[:5000] / 255. , X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# Scaling the test set as well
X_test = X_test / 255.

plt.figure(figsize= (15,15))
sns_plot  = sns.heatmap(X_valid[5], annot= True, cmap="binary")
sns_plot.figure.savefig("correlation.png")

LAYERS = [
          tf.keras.layers.Flatten(input_shape = [28,28], name="InputLayer" ),
          tf.keras.layers.Dense(400, activation="relu", name="HiddenLayer1"),
          tf.keras.layers.Dense(200, activation="relu", name = "Hiddenlayer2"),
          tf.keras.layers.Dense(10, activation="softmax", name ="OutputLater")
]

model = tf.keras.models.Sequential(LAYERS)


# Compiling model
LOSS_FUNCTION = "sparse_categorical_crossentropy"
OPTIMIZER = "SGD" 
METRICS = ["accuracy"]

model.compile(loss=LOSS_FUNCTION,
              optimizer=OPTIMIZER,
              metrics=METRICS)


# Defining callbacks
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir="logs")

CALLBACKS = [checkpoint_cb, early_stopping_cb, tensorboard_cb]


# training
EPOCHS = 10
VALIDATION_SET = (X_valid, y_valid)

history = model.fit(X_train, y_train, epochs=EPOCHS,
                    validation_data=VALIDATION_SET, batch_size=32, callbacks=CALLBACKS )             


model.evaluate(X_test, y_test)

x_new = X_test[:3]
actual = y_test[:3]
y_prob = model.predict(x_new)
y_pred = np.argmax(y_prob, axis = -1)

# plot
for data, pred, actual_data in zip(x_new, y_pred, actual):
  plt.imshow(data, cmap="binary")
  plt.title(f"Predicted {pred} and Actual {actual_data}")
  plt.axis("off")
  plt.show() 
  #plt.savefig('filename.png')
  print("######################")





