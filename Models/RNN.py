import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import LeakyReLU, Dropout, LSTM, Dense, Input, concatenate
import numpy as np
import pandas as pd
from sklearn import metrics
import seaborn as sns


x = pd.read_pickle("")
y = x[["answered_correctly"]]
pair = np.load("")  # Each pair contains the start row and end row of the unique user


### Get the past 5 attempt and current attempt ###
def get_batch(p, add):

    start = (
        p[0] + add
    )  # add = 0,1,2,3,4 so that all rows except the first 5 will be used for training
    end = p[1]

    batch_x = x.iloc[start:end, 1:]  # drop user id
    batch_y = y.iloc[start:end]
    batch_x = batch_x.to_numpy()
    batch_y = batch_y.to_numpy()

    dnn_batch = batch_x[5::5]
    batch_y = batch_y[5::5]
    dnn_batch = dnn_batch[:, 1:]

    if (batch_x.shape[0]) % 5 != 0:
        drop = (batch_x.shape[0]) % 5  # drop last few
        batch_x = batch_x[:-drop, :]
    else:
        batch_x = batch_x[:-5, :]

    batch_size = int(batch_x.shape[0] / 5)

    dnn_batch = dnn_batch.reshape((batch_size, 8))
    rnn_batch = batch_x.reshape((batch_size, 5, 9))
    batch_y = batch_y.reshape((batch_size, 1))

    return rnn_batch, dnn_batch, batch_y


def create_model():

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
learning_rate = 0.0001

model = create_model()
model.summary()


model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.AUC(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.Precision(),
    ],
)


epochs = 5
num_batch = np.shape(pair)[0] * 5
print(num_batch)  # number of batch per epoch


for epoch in range(epochs):
    count = 0
    np.random.shuffle(pair)
    for p in pair:

        for add in range(0, 5, 1):
            rnn_batch, dnn_batch, y_batch = get_batch(p, add)
            loss, acc, auc, recall, precision = model.train_on_batch(
                [rnn_batch, dnn_batch], y_batch
            )
            count = count + 1

            if count % 1000 == 0:

                print(
                    "[%.3f%% done, range: %d, epoch: %d]"
                    % (count / num_batch * 100, add, epoch)
                )
                print(
                    "[loss: %f, acc: %.3f%%, auc: %.3f%%, recall: %.3f%%, precision: %.3f%%]"
                    % (loss, 100 * acc, 100 * auc, 100 * recall, 100 * precision)
                )


model.save_weights("")

del x
del y

test = pd.read_pickle("")
x = test.drop(columns=["lecture_count"])
y = x[["answered_correctly"]]

test_pair = np.load("")

y_pred = [[]]
y_val = [[]]

for p in test_pair:
    for add in range(0, 5, 1):
        rnn_batch, dnn_batch, y_val_batch = get_batch(p, add)
        y_pred_batch = model.predict_on_batch([rnn_batch, dnn_batch])
        y_pred = np.append(y_pred, y_pred_batch)
        y_val = np.append(y_val, y_val_batch)


np.save("", y_pred)
np.save("", y_val)


y_pred = np.where(y_pred > 0.5, 1, 0)
accuracy = metrics.accuracy_score(y_val, y_pred)
precision = metrics.precision_score(y_val, y_pred)

recall = metrics.recall_score(y_val, y_pred)
cm = metrics.confusion_matrix(y_val, y_pred)

auc = metrics.roc_auc_score(y_val, y_pred)

print("Accuracy: %.3f%%" % (accuracy * 100.0))
print("Precision: %.3f%%" % (precision))
print("Recall: %.3f%%" % (recall))

print("AUC: %.3f%%" % (auc))

disp = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
disp.plot()
plt.show()