from imports import *
from load import load_data
from load import em

class DropConnect(tf.keras.layers.Layer):
    def __init__(self, p):
        super(DropConnect, self).__init__()
        self.p = p
        self.dropout = tf.keras.layers.Dropout(1-self.p)

    def call(self, inputs, training=False):
        if not training:
            return inputs
        return self.dropout(inputs) * self.p
def getmodel():
    model = tf.keras.models.Sequential([tf.keras.layers.InputLayer((INPUT_WIDTH, INPUT_HEIGHT, CHANNELS), name='input_layer')])

    psz64 = 2
    psz32 = 2
    fsz32 = 5
    fsz64 = 3
    ncv32 = 2
    ncv64 = 2
    for i in range(ncv32):
        model.add(tf.keras.layers.Conv2D(32, (fsz32, fsz32), activation='relu', padding = 'same', name="conv32_{}".format(i)))
        model.add(tf.keras.layers.BatchNormalization(name = "bn32_{}".format(i)))
    if ncv32 > 0:
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(psz32, psz32)))
    for i in range(ncv64):
        model.add(tf.keras.layers.Conv2D(64, (fsz64, fsz64), activation='relu', padding = 'same'))
        model.add(tf.keras.layers.BatchNormalization(name = "bn64_{}".format(i)))
    if ncv64 > 0:
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(psz64, psz64)))
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization(name = "bn_dense"))
    model.add(DropConnect(0.6))

    model.add(tf.keras.layers.Dense(10))

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(optimizer= optimizer,
                   loss=loss_fn,
                   metrics=['accuracy'])
    return model
