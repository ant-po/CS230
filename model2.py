# Model 2 - single LSTM layer

from keras.layers import Input, Dense, LSTM, concatenate
from keras.models import Model
from keras.callbacks import TensorBoard
from build_dataset import readDataFromCsv, pickBest
from matplotlib import pyplot
from time import time
import numpy as np
import pandas as pd

# user parameters
num_lstm_units = [10]
num_dense_units = [3]
dense_activation ='sigmoid'
num_epochs = 200
batch_size = 64

# log results in /logs folder to be viewed via TensorBoard
tensorboard = TensorBoard(log_dir="./logs/{}".format(time()))

# fetch the raw train/test datasets
x_train, y_train, x_test, y_test = readDataFromCsv('./data/processed_data/latest_dataset')

# process the train/test datasets
num_assets = y_train.shape[1]
num_timesteps = np.int(x_train.shape[1] / num_assets)
x1_train = x_train[:, 0:num_timesteps]
x1_train = np.reshape(x1_train, (x1_train.shape[0], x1_train.shape[1], 1))
x2_train = x_train[:, num_timesteps:num_timesteps*2]
x2_train = np.reshape(x2_train, (x2_train.shape[0], x2_train.shape[1], 1))
x3_train = x_train[:, num_timesteps*2:]
x3_train = np.reshape(x3_train, (x3_train.shape[0], x3_train.shape[1], 1))
x1_train = np.append(x1_train, x2_train, axis=-1)
x1_train = np.append(x1_train, x3_train, axis=-1)

x1_test = x_test[:, 0:num_timesteps]
x1_test = np.reshape(x1_test, (x1_test.shape[0], x1_test.shape[1], 1))
x2_test = x_test[:, num_timesteps:num_timesteps*2]
x2_test = np.reshape(x2_test, (x2_test.shape[0], x2_test.shape[1], 1))
x3_test = x_test[:, num_timesteps*2:]
x3_test = np.reshape(x3_test, (x3_test.shape[0], x3_test.shape[1], 1))
x1_test = np.append(x1_test, x2_test, axis=-1)
x1_test = np.append(x1_test, x3_test, axis=-1)

y_train_rank = pd.DataFrame(y_train).rank(axis=1, method='first', ascending=True)
y_train_final = pickBest(y_train_rank.values)
y_test_rank = pd.DataFrame(y_test).rank(axis=1, method='first', ascending=True)
y_test_final = pickBest(y_test_rank.values)

# define the model
input = Input(shape=(num_timesteps, num_assets), dtype='float32')
X = input
for i in range(0, len(num_lstm_units)):
    X = LSTM(units=num_lstm_units[i])(X)
for i in range(0, len(num_dense_units)):
    X = Dense(units=num_dense_units[i], activation=dense_activation)(X)
out = Dense(units=3, activation='softmax')(X)

# compile and train the model
model = Model(inputs=input, outputs=out)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
history = model.fit(x1_train, y_train_final, epochs=num_epochs, batch_size=batch_size, verbose=1,
          validation_data=(x1_test, y_test_final), callbacks=[tensorboard], shuffle=True)

# plot train/test loss and accuracy for each epoch
pyplot.plot(history.history['loss'], label='train_loss')
pyplot.plot(history.history['val_loss'], label='test_loss')
pyplot.legend()
pyplot.plot(history.history['categorical_accuracy'], label='train_accuracy')
pyplot.plot(history.history['val_categorical_accuracy'], label='test_accuracy')
pyplot.legend()
pyplot.show()
