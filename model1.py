# Model 1 - stacked fully-connected layers

from keras.layers import Input, Dense, BatchNormalization, Dropout, Activation
from keras.models import Model
from keras.callbacks import TensorBoard
from build_dataset import readDataFromCsv, pickBest
from matplotlib import pyplot
from time import time
import pandas as pd
import numpy as np

# user parameters
num_hid_units = [10, 10, 10]
dense_activation ='relu'
do_rate = 0.5
num_epochs = 200
batch_size = 64

# log results in /logs folder to be viewed via TensorBoard
tensorboard = TensorBoard(log_dir="./logs/{}".format(time()))

# fetch the raw train/test datasets
x_train, y_train, x_test, y_test = readDataFromCsv('./data/processed_data/latest_dataset')

# process the train/test datasets
y_train_rank = pd.DataFrame(y_train).rank(axis=1, method='first', ascending=True)
y_train_final = pickBest(y_train_rank.values)
y_test_rank = pd.DataFrame(y_test).rank(axis=1, method='first', ascending=True)
y_test_final = pickBest(y_test_rank.values)
x_shape = x_train.shape[1]

# define the model
input = Input(shape=(x_shape,), dtype='float32')
X = input
for i in range(0, len(num_hid_units)):
    X = Dense(units=num_hid_units[i], input_shape=(x_shape, 1), activation=None)(X)
    X = BatchNormalization()(X)
    X = Activation(dense_activation)(X)
    X = Dropout(do_rate)(X)
out = Dense(units=3, activation='softmax')(X)

# compile and train the model
model = Model(inputs=input, outputs=out)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(x_train, y_train_final, epochs=num_epochs, batch_size=batch_size, verbose=1, shuffle=True,
          validation_data=[x_test, y_test_final], callbacks=[tensorboard])

# evaluate test set accuracy
acc = model.evaluate(x_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], acc[1]*100))

# plot train/test loss and accuracy for each epoch
pyplot.plot(history.history['loss'], label='train_loss')
pyplot.plot(history.history['val_loss'], label='test_loss')
pyplot.legend()
pyplot.plot(history.history['categorical_accuracy'], label='train_accuracy')
pyplot.plot(history.history['val_categorical_accuracy'], label='test_accuracy')
pyplot.legend()
pyplot.show()
