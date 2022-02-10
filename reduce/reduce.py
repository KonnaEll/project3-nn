import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import History, EarlyStopping

tf.executing_eagerly()


class CustomMinMaxScaler:
    def __init__(self, x):
        self.min_ = x.min()
        self.max_ = x.max()

    def scale(self, x):
        return (x - self.min_) / (self.max_ - self.min_)

    def inverse_scale(self, x):
        return x * (self.max_ - self.min_) + self.min_


argparser = argparse.ArgumentParser()
argparser.add_argument('-d', help='Dataset', default='dataset.csv')
argparser.add_argument('-q', help='Query', default='query.csv')
argparser.add_argument('-od', help='Dataset_reduced', default='dataset_reduced.csv')
argparser.add_argument('-oq', help='Query_reduced', default='query_reduced.csv')
args = argparser.parse_args()

dataset = args.d
query = args.q
dataset_reduced = args.od
query_reduced = args.oq

# Hyperparameters
window = 10
max_epochs = 20
batch_size = 32
dropout = 0.2

experiment_folder = 'experiments_optimal'
os.makedirs(experiment_folder, exist_ok=True)

# Read dataset
df = pd.read_csv(dataset, header=None, sep="\t")
datanames = df.iloc[:, 0].values
df = df.drop(columns=[0])
data = df.values
# Read query
query = pd.read_csv(query, header=None, sep="\t")
querynames = query.iloc[:, 0].values
query = query.drop(columns=[0])
query = query.values

num_columns = len(df.columns)

# Min-Max scaler wste na metatrepsoume oles tis times sto [0, 1]
scaler = CustomMinMaxScaler(data)
train_set_scaled = scaler.scale(data)
test_set_scaled = scaler.scale(query)


trainX = np.reshape(train_set_scaled, (-1, num_columns // window, window))
testX = np.reshape(test_set_scaled, (-1, num_columns // window, window))


print('Neural network input shape')
print(f'Train shape: {trainX.shape}')

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, padding='same', input_shape=(trainX.shape[1], window)))
model.add(Dropout(dropout))
model.add(Conv1D(filters=32, kernel_size=3, padding='same'))
model.add(Dropout(dropout))
model.add(Conv1D(filters=3, kernel_size=3, padding='same'))
model.add(Dropout(dropout))
model.add(Conv1D(filters=32, kernel_size=3, padding='same'))
model.add(Dropout(dropout))
model.add(Conv1D(filters=10, kernel_size=3, padding='same'))
print(model.summary())

# Training
model.compile(optimizer='adam', loss='mse')
history = History()
# Epilegoume tyxaia 10% tou training set ws validation, gia na mporoume na apofygoume to overfitting
# Early stopping, dhladh an se 3 epoxes den exei veltiwthei to validation loss, stamatame to training
hist = model.fit(trainX, trainX,
                 epochs=max_epochs,
                 batch_size=batch_size,
                 validation_split=0.1,
                 callbacks=[history])
hist = hist.history

plt.figure()
plt.plot(hist['loss'], color='red', label='Train loss')
plt.plot(hist['val_loss'], color='blue', label='Validation loss')
plt.savefig(os.path.join(experiment_folder, 'training_history.png'))

latent_projection = keras.Model(inputs=model.input, outputs=model.get_layer('conv1d_2').output)
train_reduced = latent_projection.predict(trainX)
test_reduced = latent_projection.predict(testX)

train_reduced = pd.DataFrame(np.reshape(train_reduced, (train_reduced.shape[0], -1)))
train_reduced.insert(0, 'name', datanames)
train_reduced.to_csv(dataset_reduced, sep='\t', index=False, header=False)
test_reduced = pd.DataFrame(np.reshape(test_reduced, (test_reduced.shape[0], -1)))
test_reduced.insert(0, 'name', querynames)
test_reduced.to_csv(query_reduced, sep='\t', index=False, header=False)
