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
    return check2()
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

def check2():
    inputs = tf.keras.Input((INPUT_WIDTH, INPUT_HEIGHT, CHANNELS))

    conv = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=(2,2))
    bn = tf.keras.layers.BatchNormalization()
    x = conv(inputs)
    x1 = bn(x)
    conv = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')
    bn = tf.keras.layers.BatchNormalization()
    x = conv(x1)
    x = bn(x)
    conv = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')
    bn = tf.keras.layers.BatchNormalization()
    x = conv(x)
    x2 = bn(x)

    x = x2
    # x = tf.keras.layers.Add()([x1, x2])

    conv = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=(2,2))
    bn = tf.keras.layers.BatchNormalization()
    x = conv(x)
    x1 = bn(x)
    conv = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')
    bn = tf.keras.layers.BatchNormalization()
    x = conv(x1)
    x = bn(x)
    conv = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')
    bn = tf.keras.layers.BatchNormalization()
    x = conv(x)
    x2 = bn(x)

    x = x2
    # x = tf.keras.layers.Add()([x1, x2])

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = DropConnect(0.6)(x)
    outputs = tf.keras.layers.Dense(10, activation='relu')(x)

    model = tf.keras.Model(inputs = inputs, outputs = outputs)
    model.summary()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.RMSprop(lr=INITIAL_LR, rho=0.9, epsilon=1e-08, decay=0.1)
    optimizer = tf.keras.optimizers.Adam(INITIAL_LR)
    optimizer = tf.keras.optimizers.SGD(INITIAL_LR, decay=5e-4)
    model.compile(optimizer= optimizer,
                   loss=loss_fn,
                   metrics=['accuracy', get_lr_metric(optimizer)])
    return model
def getres():
    x = tf.keras.layers.InputLayer((INPUT_WIDTH, INPUT_HEIGHT, CHANNELS), name='input_layer')

    b1 = tf.keras.layers.Conv2D(6, (3, 3), activation='relu', padding='same', strides=(2,2))
    b1 = tf.keras.layers.BatchNormalization()(b1)
    b11 = tf.keras.layers.Conv2D(6, (3, 3), activation='relu', padding='same')(b1)
    b11 = tf.keras.layers.BatchNormalization()(b11)
    b11 = tf.keras.layers.Conv2D(6, (3, 3), activation='relu', padding='same')(b11)
    b11 = tf.keras.layers.BatchNormalization()(b11)

    b2 = tf.keras.layers.Conv2D(6, (3, 3), activation='relu', padding='same', strides=(2,2))
    b2 = tf.keras.layers.BatchNormalization()(b2)
    b21 = tf.keras.layers.Conv2D(6, (3, 3), activation='relu', padding='same')(b2)
    b21 = tf.keras.layers.BatchNormalization()(b21)
    b21 = tf.keras.layers.Conv2D(6, (3, 3), activation='relu', padding='same')(b21)
    b21 = tf.keras.layers.BatchNormalization()(b21)

    f = tf.keras.layers.Flatten()
    d1 = tf.keras.layers.Dense(64, activation='relu')
    d1 = DropConnect(0.6)(d1)
    d2 = tf.keras.layers.Dense(64, activation='relu')
    d2 = DropConnect(0.6)(d2)

    d3 = tf.keras.layers.Dense(10, activation='relu')

    class Model(tf.keras.Model):
        def __init__(self):
            super(Model, self).__init__()
        def call(self, inputs, training=False):
            return d3(d2(d1(f(b21(b2(b11(b1(x(inputs)))))))))
    model = Model()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(optimizer= optimizer,
                   loss=loss_fn,
                   metrics=['accuracy'])
    return model
def getsmall():
    model = tf.keras.models.Sequential([tf.keras.layers.InputLayer((INPUT_WIDTH, INPUT_HEIGHT, CHANNELS), name='input_layer')])

    m1 = tf.keras.models.Sequential([tf.keras.layers.InputLayer((INPUT_WIDTH, INPUT_HEIGHT, CHANNELS), name='input_layer1')])

    m1.add(tf.keras.layers.Conv2D(6, (3, 3), activation='relu', padding='same', strides=(2,2)))
    m1.add(tf.keras.layers.BatchNormalization())
    m1.add(tf.keras.layers.Conv2D(6, (3, 3), activation='relu', padding='same'))
    m1.add(tf.keras.layers.BatchNormalization())
    m1.add(tf.keras.layers.Conv2D(6, (3, 3), activation='relu', padding='same'))
    m1.add(tf.keras.layers.BatchNormalization())

    m2 = tf.keras.models.Sequential([tf.keras.layers.InputLayer((INPUT_WIDTH//2, INPUT_HEIGHT//2, 6), name='input_layer2')])
    m2.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=(2,2)))
    m2.add(tf.keras.layers.BatchNormalization())
    m2.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    m2.add(tf.keras.layers.BatchNormalization())
    m2.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    m2.add(tf.keras.layers.BatchNormalization())

    model.add(m1)
    model.add(m2)
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(DropConnect(0.6))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(DropConnect(0.6))

    model.add(tf.keras.layers.Dense(10))

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(optimizer= optimizer,
                   loss=loss_fn,
                   metrics=['accuracy'])
    return model

def getbig():
    model = tf.keras.models.Sequential([tf.keras.layers.InputLayer((INPUT_WIDTH, INPUT_HEIGHT, CHANNELS), name='input_layer')])

    psz32, psz64 = 2, 2
    fsz32, fsz64 = 5, 3
    ncv32, ncv64 = 2, 2
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
