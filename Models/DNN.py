import tensorflow as tf
from keras.layers import LeakyReLU
import pandas as pd


train = pd.read_pickle("")

y_train = train["answered_correctly"]
x_train = train.drop(columns=["answered_correctly", "user_id"])
del train


class LearningRateReducerCb(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        old_lr = self.model.optimizer.lr.read_value()
        new_lr = old_lr * 0.9
        print(
            "\nEpoch: {}. Reducing Learning Rate from {} to {}".format(
                epoch, old_lr, new_lr
            )
        )
        self.model.optimizer.lr.assign(new_lr)


def create_model(my_learning_rate, input_size):

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


learning_rate = 0.001
epochs = 13
batch_size = 128

model = create_model(learning_rate, 8)
model.summary()


history = model.fit(
    x_train,
    y_train,
    callbacks=[LearningRateReducerCb()],
    batch_size=batch_size,
    epochs=epochs,
)


model.save_weights("")
