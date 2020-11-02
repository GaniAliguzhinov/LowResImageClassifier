from imports import *

MEAN = 0.5
STD = 0.2
def load_data():
    if not FROM_FILE:
        (Xtrain, Ytrain), (Xtest, Ytest) = tf.keras.datasets.mnist.load_data()
        Xtrain, Xtest = Xtrain.reshape(-1, INPUT_WIDTH, INPUT_HEIGHT, CHANNELS), \
                        Xtest.reshape(-1, INPUT_WIDTH, INPUT_HEIGHT, CHANNELS)
        return Xtrain, Ytrain, Xtest, Ytest
    else:
        train = pd.read_csv(TRAIN_FILE)
        test = pd.read_csv(TEST_FILE)
        Xtrain = train.values[:,1:].reshape(-1, INPUT_WIDTH, INPUT_HEIGHT, CHANNELS)
        Xtest = test.values.reshape(-1, INPUT_WIDTH, INPUT_HEIGHT, CHANNELS)
        Ytrain = train.values[:,0].reshape(-1)
        Ytest = np.zeros((test.values.shape[0]))

        return Xtrain, Ytrain, Xtest, Ytest
def mapValues(values, domain_l, domain_h, codomain_l=0.0, codomain_h=1.0, save=False):
    global MEAN
    global STD
    l, h, L, H = domain_l, domain_h, codomain_l, codomain_h
    v = (values - l) * (H - L)/(h - l) + L
    if save:
        MEAN = v.mean().astype(np.float32)
        STD = v.std().astype(np.float32)
    return v

def normalize(x):
    m = MEAN
    s = STD
    return (x-m)/s
def em(x):
    return x
def getgen():
    return tf.keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0.1, # Randomly zoom image
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            shear_range=0.4,      # randomly apply shearing in range (degrees, 0 to 180)
            vertical_flip=False)  # randomly flip images
