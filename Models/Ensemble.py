from sklearn import ensemble
import tensorflow as tf
import pandas as pd
import numpy as np
from keras import backend as K
from keras.layers import LeakyReLU, Dropout, LSTM, Dense, Input, concatenate
from keras.models import Model


class LearningRateReducerCb(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        old_lr = self.model.optimizer.lr.read_value()
        new_lr = old_lr * 0.8
        print(
            "\nEpoch: {}. Reducing Learning Rate from {} to {}".format(
                epoch, old_lr, new_lr
            )
        )
        self.model.optimizer.lr.assign(new_lr)


def create_DNN(my_learning_rate, input_size):

    model = tf.keras.models.Sequential()

    model.add(
        tf.keras.layers.Dense(
            units=128,
            input_shape=(input_size,),
            name="Hidden1",
        )
    )
    model.add(LeakyReLU(alpha=0.1))
    model.add(tf.keras.layers.Dropout(0.05))

    model.add(
        tf.keras.layers.Dense(
            units=128,
            kernel_initializer="HeNormal",
            name="Hidden2",
        )
    )
    model.add(LeakyReLU(alpha=0.1))
    model.add(tf.keras.layers.Dropout(0.05))

    model.add(
        tf.keras.layers.Dense(
            units=128,
            kernel_initializer="HeNormal",
            name="Hidden3",
        )
    )
    model.add(LeakyReLU(alpha=0.1))
    model.add(tf.keras.layers.Dropout(0.05))

    model.add(
        tf.keras.layers.Dense(
            units=128,
            kernel_initializer="HeNormal",
            name="Hidden4",
        )
    )
    model.add(LeakyReLU(alpha=0.1))
    model.add(tf.keras.layers.Dropout(0.05))

    model.add(
        tf.keras.layers.Dense(
            units=64,
            kernel_initializer="HeNormal",
            name="Hidden5",
        )
    )
    model.add(LeakyReLU(alpha=0.1))
    model.add(tf.keras.layers.Dropout(0.05))

    # Define the output layer.
    model.add(
        tf.keras.layers.Dense(
            units=1,
            kernel_initializer="GlorotNormal",
            activation="sigmoid",
            name="Output",
        )
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=my_learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.AUC(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.Precision(),
        ],
    )

    return model


def create_RNN():

    input1 = Input(shape=rnn_input)
    input2 = Input(shape=dnn_input)

    lstm1 = LSTM(32, return_sequences=True)(input1)
    Dropout1 = Dropout(0.2)(lstm1)

    lstm2 = LSTM(32)(Dropout1)
    Dropout2 = Dropout(0.2)(lstm2)

    merged = concatenate([Dropout2, input2])

    dense1 = Dense(128)(merged)
    lr1 = LeakyReLU(alpha=0.1)(dense1)
    Dropout3 = Dropout(0.2)(lr1)

    dense2 = Dense(64)(Dropout3)
    lr2 = LeakyReLU(alpha=0.1)(dense2)
    Dropout4 = Dropout(0.2)(lr2)

    output = Dense(1, activation="sigmoid", name="Output")(Dropout4)

    model = Model([input1, input2], output)

    return model


dnn_input = 8
rnn_input = (5, 9)

input_shape = 4
num_classes = 1
learning_rate = 0.001
epochs = 5
batch_size = 32

# load the trained models

DNN1 = create_DNN(learning_rate, 8)
DNN1.load_weights("")
DNN2 = create_DNN(learning_rate, 8)
DNN2.load_weights("")
RNN1 = create_RNN()
RNN1.load_weights("")
RNN2 = create_RNN()
RNN2.load_weights("")

# use 10000 students from each dataset
p_train1 = np.load("")
last_u1 = p_train1[9999, 0]
p_train1 = p_train1[:10000, :]

p_train2 = np.load("")
last_u2 = p_train2[9999, 0]
p_train2 = p_train2[:10000, :]

p_train2 = p_train2 + last_u1
p_train = np.vstack((p_train1, p_train2))

df = pd.read_pickle("")
train1 = df[:last_u1]
del df

df = pd.read_pickle("")
train2 = df[:last_u2]
del df

x = pd.concat([train1, train2], ignore_index=True)

y = x[["answered_correctly"]]


print("done reading data")


def get_batch(p, add):

    start = p[0] + add
    end = p[1]

    batch_x = x.iloc[start:end, 1:]
    batch_y = y.iloc[start:end]
    batch_x = batch_x.to_numpy()
    batch_y = batch_y.to_numpy()

    dnn_batch = batch_x[5::5]
    dnn_batch = dnn_batch[:, 1:]

    if (batch_x.shape[0]) % 5 != 0:
        drop = (batch_x.shape[0]) % 5
        batch_x = batch_x[:-drop, :]
    else:
        batch_x = batch_x[:-5, :]

    batch_size = int(batch_x.shape[0] / 5)

    batch_y = batch_y[5::5]
    dnn_batch = dnn_batch.reshape((batch_size, 8))
    rnn_batch = batch_x.reshape((batch_size, 5, 9))
    batch_y = batch_y.reshape((batch_size, 1))

    return rnn_batch, dnn_batch, batch_y


def get_input(rnn_b, dnn_b):
    rnn1 = RNN1.predict_on_batch([rnn_b, dnn_b])
    rnn2 = RNN2.predict_on_batch([rnn_b, dnn_b])
    dnn1 = DNN1.predict_on_batch(dnn_b)
    dnn2 = DNN2.predict_on_batch(dnn_b)
    input = np.hstack((rnn1, rnn2, dnn1, dnn2))
    return input


y_pred = np.array([]).reshape(0, 4)
y_actual = np.array([]).reshape(0, 1)

for p in p_train:
    for add in range(0, 5, 1):
        rnn_batch, dnn_batch, y_actual_batch = get_batch(p, add)
        y_pred_batch = get_input(rnn_batch, dnn_batch)
        y_pred = np.vstack((y_pred, y_pred_batch))
        y_actual = np.vstack((y_actual, y_actual_batch))


np.save("", y_pred)
np.save("", y_actual)

print("finished prediction")


def create_model(my_learning_rate, input_size, num_classes):

    model = tf.keras.models.Sequential()

    model.add(
        tf.keras.layers.Dense(
            units=6,
            input_shape=(input_size,),
            activation="relu",
            name="Hidden1",
        )
    )

    model.add(
        tf.keras.layers.Dense(
            units=2,
            activation="relu",
            name="Hidden2",
        )
    )

    model.add(
        tf.keras.layers.Dense(units=num_classes, activation="sigmoid", name="Output")
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=my_learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.AUC(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.Precision(),
        ],
    )

    return model


ensemble = create_model(learning_rate, input_shape, num_classes)

ensemble.fit(
    y_pred,
    y_actual,
    callbacks=[LearningRateReducerCb()],
    batch_size=batch_size,
    epochs=epochs,
)

ensemble.save_weights("")