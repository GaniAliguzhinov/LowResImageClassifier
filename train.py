#================================================================================
# IMPORTS
#================================================================================
import os
import silence_tensorflow.auto
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
import itertools
import pandas as pd
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns

from keras.callbacks import EarlyStopping
#!pip install -q -U keras-tuner
import kerastuner as kt
#================================================================================
# Global Constants
#================================================================================
TRAIN_FILE = '../input/digit-recognizer/train.csv'
TEST_FILE = '../input/digit-recognizer/test.csv'
FROM_FILE = False
INPUT_WIDTH = 28
INPUT_HEIGHT = 28
CHANNELS = 1
SHOW_PLOTS = True
BS = 64
SCHEDULE = [28, 8, 4]
DO_TRAIN = True
INITIAL_LR = 0.01
DO_TUNE = False
DO_EARLY_STOP = False
WEIGHTS_NAME = 'MODEL_WEIGHTS'


#================================================================================
# LOADING DATA 
#================================================================================

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
        
Xall, Yall, Xtest, Ytest = load_data()
Xall, Xtest = Xall.astype(np.float32), Xtest.astype(np.float32)
Nall, Ntest = Xall.shape[0], Xtest.shape[0]
print("Training data before split: {}\nTest data: {}\nTraining labels before split: {}\nTest labels: {}".
                                              format(Xall.shape, Xtest.shape, Yall.shape, Ytest.shape))
#================================================================================
# CHECK THE DATASET FOR LABEL DISTRIBUTION AND CORRUPTED DATA
#================================================================================

columns = ['digit']
index = [f'index_{num}' for num in range(Nall)]
YallDf = pd.DataFrame(Yall, columns=columns, index=index)
if SHOW_PLOTS:
    g = sns.countplot(YallDf['digit'])
    g.get_figure().savefig('trainingClassDistribution.png')
    
print("Count of corrupted images in the training set:\n{}\n\nCount of corrupted images in the testing set:\n{}".
          format(np.sum(np.any(np.isnan(Xall.reshape(Nall, -1)), axis=0)), 
                 np.sum(np.any(np.isnan(Xtest.reshape(Ntest, -1)), axis=0))))
                 
#================================================================================
# MAPPING VALUES TO [0, 1]
#================================================================================
def mapValues(values, domain_l, domain_h, codomain_l=0.0, codomain_h=1.0):
    l, h, L, H = domain_l, domain_h, codomain_l, codomain_h
    return (values - l) * (H - L)/(h - l) + L
Xall, Xtest = mapValues(Xall, 0, 255), mapValues(Xtest, 0, 255)
print("Intensities after scaling: min={}, max={}".format(np.min(Xall.flatten()), np.max(Xall.flatten())))

if SHOW_PLOTS:
    data = np.array([Xall[:Nall//4].reshape(Xall.shape[0]//4*INPUT_WIDTH, INPUT_HEIGHT),
                     Xtest[:Ntest//4].reshape(Xtest.shape[0]//4*INPUT_WIDTH, INPUT_HEIGHT)])
    xaxes = ['intensity','intensity']
    yaxes = ['count','count']
    titles = ['Histogram: Training set','Histogram: Test set'] 

    f,a = plt.subplots(nrows=2, figsize=[10,6])
    a = a.ravel()
    for idx,ax in enumerate(a):
        ax.hist(data[idx], bins = [0,0.2,0.4,0.6,0.8,1])
        ax.set_title(titles[idx])
        ax.set_xlabel(xaxes[idx])
        ax.set_ylabel(yaxes[idx])
    plt.tight_layout()
    plt.show()
    plt.savefig('intensitiesHist.png')
    
#================================================================================
# SHOWING TRAINING EXAMPLES
#================================================================================
if SHOW_PLOTS:
    nrows, ncols = 5, 5  # array of sub-plots
    figsize = [10, 9]     # figure size, inches
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for i, axi in enumerate(ax.flat):
        imageId = np.random.randint(0, Nall)
        axi.imshow(Xall[imageId].reshape(INPUT_WIDTH, INPUT_HEIGHT))    
        axi.set_title("Label: {}".format(Yall[imageId]))
    plt.tight_layout(False)
    print("Some train images")
    plt.show()  
    plt.savefig('trainingImages.png')
    
#================================================================================
# SHOWING TESTING EXAMPLES
#================================================================================
if SHOW_PLOTS:
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for i, axi in enumerate(ax.flat):
        imageId = np.random.randint(0, Xtest.shape[0])
        axi.imshow(Xtest[imageId].reshape(INPUT_WIDTH, INPUT_HEIGHT))    
        axi.set_title("Label: {}".format(Ytest[imageId]))
    plt.tight_layout(False)
    print("Some test images")
    plt.show()
    plt.savefig('testingImages.png')
    
#================================================================================
# SPLIT TRAINING DATA
#================================================================================
Xtrain, Xval, Ytrain, Yval = train_test_split(Xall, Yall, test_size=0.1, random_state = 13)
print("Train data: {}, Validation data: {}".format(Xtrain.shape, Xval.shape))

#================================================================================
# DATA AUGMENTATION AND BATCHES
#================================================================================

gen = tf.keras.preprocessing.image.ImageDataGenerator(
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

batches = gen.flow(Xtrain, Ytrain, batch_size=BS)

#================================================================================
# METHODS USED BY THE NETWORK
#================================================================================
seed = 13
np.random.seed(seed)

def normalize(x, m=Xall.mean().astype(np.float32), s=Xall.std().astype(np.float32)):
    return (x-m)/s

class DropConnect(tf.keras.layers.Layer):
    def __init__(self, p):
        super(DropConnect, self).__init__()
        self.p = p
        self.dropout = tf.keras.layers.Dropout(1-self.p)
    
    def call(self, inputs, training=False):
        if not training:
            return inputs
        return self.dropout(inputs) * self.p

def scheduler(epoch, lr, INIT_LR=INITIAL_LR, schedule=SCHEDULE):
    if epoch < schedule[0]:
        return INIT_LR 
    elif epoch < schedule[0] + schedule[1]*1:
        return INIT_LR * 0.5
    elif epoch < schedule[0] + schedule[1]*2:
        return INIT_LR * 0.1
    elif epoch < schedule[0] + schedule[1]*2 + schedule[2]:
        return INIT_LR * 0.05
    elif epoch < schedule[0] + schedule[1]*2 + schedule[2]*2:
        return INIT_LR * 0.01
    elif epoch < schedule[0] + schedule[1]*2 + schedule[2]*3:
        return INIT_LR * 0.005
    else:
        return INIT_LR * 0.001

learning_rate_reduction = tf.keras.callbacks.LearningRateScheduler(scheduler)

earlystopper = EarlyStopping(monitor='val_loss', min_delta=0,
                             patience=3, verbose=1, mode='auto')
callbacks = [learning_rate_reduction]
if DO_EARLY_STOP:
    callbacks += [earlystopper]
#================================================================================
# MODEL DEFINITION
#================================================================================
keep_p_last = 0.5
keep_p_conv = 0.75

def model_builder(hp):
    model2 = tf.keras.models.Sequential([tf.keras.layers.Lambda(normalize, input_shape=(INPUT_WIDTH,INPUT_HEIGHT,CHANNELS))])
    
    
    
    if DO_TUNE:
        fsz32 = hp.Int('fsz32', min_value = 2, max_value = 7, step = 1)
        fsz64 = hp.Int('fsz64', min_value = 2, max_value = 7, step = 1)
        psz32 = hp.Int('psz32', min_value = 2, max_value = 5, step = 1)
        psz64 = hp.Int('psz64', min_value = 2, max_value = 5, step = 1)
    else:
        fsz32 = 5
        psz32 = 2
        fsz64 = 3
        psz64 = 2

    model2.add(tf.keras.layers.Conv2D(32, (fsz32, fsz32), activation='relu'))
    model2.add(tf.keras.layers.BatchNormalization())
    model2.add(tf.keras.layers.Conv2D(32, (fsz32, fsz32), activation='relu'))
    model2.add(tf.keras.layers.BatchNormalization())    
    model2.add(tf.keras.layers.MaxPooling2D(pool_size=(psz32, psz32)))
    
#     hp_p_conv = hp.Float('keep_p_conv', min_value = 0.05, max_value = 1.0, step = 0.05)
#     model2.add(DropConnect(hp_p_conv))

    model2.add(tf.keras.layers.Conv2D(64, (fsz64, fsz64), activation='relu'))
    model2.add(tf.keras.layers.BatchNormalization())
    model2.add(tf.keras.layers.Conv2D(64, (fsz64, fsz64), activation='relu'))
    model2.add(tf.keras.layers.BatchNormalization())

    model2.add(tf.keras.layers.MaxPooling2D(pool_size=(psz64, psz64)))
    
#     model2.add(DropConnect(hp_p_conv))

    model2.add(tf.keras.layers.Flatten())
    
    model2.add(tf.keras.layers.Dense(256, activation='relu'))
    model2.add(tf.keras.layers.BatchNormalization())
    
#     hp_p_last = hp.Float('keep_p_last', min_value = 0.05, max_value = 1.0, step = 0.05)  
    model2.add(DropConnect(0.6))

    model2.add(tf.keras.layers.Dense(10))
    
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)
    model2.compile(optimizer= optimizer,
                   loss=loss_fn,
                   metrics=['accuracy'])
    return model2

#================================================================================
# TUNE HYPER PARAMETERS
#================================================================================

if DO_TUNE:
    tuner = kt.Hyperband(model_builder,
                         objective = 'val_accuracy', 
                         max_epochs = SCHEDULE[0] + 2*SCHEDULE[1] + 4*SCHEDULE[2],
                         factor = 3,
                         directory = 'my_dir',
                         project_name = 'MNIST_TUNER')   

    tuner.search(Xtrain, Ytrain, epochs = SCHEDULE[0] + 2*SCHEDULE[1] + 4*SCHEDULE[2], validation_data = (Xval, Yval), 
                 callbacks = [learning_rate_reduction, earlystopper])

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

    print(f"""
    Optimal filter sizes: {best_hps.get('fsz32')} for convolution 32, and {best_hps.get('fsz64')} for convolution 64.
    Optimal MaxPool sizes: {best_hps.get('psz32')} after convolution 32, and {best_hps.get('psz64')} after convolution 64.
    """)
    model = tuner.hypermodel.build(best_hps)
    tuner.results_summary()
    tuner.search_space_summary()
else:
    model = model_builder(1)



#================================================================================
# TRAIN
#================================================================================

if DO_TRAIN:
    history=model.fit_generator(generator = batches, 
                             steps_per_epoch = int(Xtrain.shape[0]/BS), 
                             epochs = SCHEDULE[0] + 2*SCHEDULE[1] + 4*SCHEDULE[2],
                             validation_data = (Xval, Yval),
                            verbose = 2,
                             callbacks = callbacks)
else:
    model.load_weights(WEIGHTS_NAME)
print("Model performance on the validation set:")
model.evaluate(Xval, Yval, verbose=2)

#================================================================================
# TRAINING AND VALIDATION LOSS AND ACCURACY CURVES
#================================================================================
if SHOW_PLOTS:
    fig, ax = plt.subplots(2,1)
    ax[0].plot(history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
    ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)
    plt.savefig('lossAccuracyCurves.png')

#================================================================================
# CONFUSION MATRIX
#================================================================================

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusionMatrix.png')

if SHOW_PLOTS:
    Ypred = np.argmax(tf.nn.softmax(model.predict(Xval)), axis = 1)

    confusion_mtx = confusion_matrix(Yval, Ypred) 
    # plot the confusion matrix
    plot_confusion_matrix(confusion_mtx, classes = range(10)) 

#================================================================================
# SHOWING WRONG PREDICTIONS
#================================================================================
Ypred_probs = tf.nn.softmax(model.predict(Xval)).numpy()
Ypred = np.argmax(Ypred_probs, axis = 1)

Ytrue_probs = to_categorical(Yval).astype(np.float32)
Ytrue = Yval

ErrorMask = Ytrue != Ypred
Ids_true =  np.argmax(Ytrue_probs[ErrorMask], axis=1)

ProbsOfTrue = Ypred_probs[ErrorMask][np.arange(len(Ids_true)), Ids_true]

if SHOW_PLOTS:
    nrows, ncols = min(4, np.sum(ErrorMask)//4), 4
    ids = np.argsort(ProbsOfTrue)[:ncols*nrows]
    figsize = [10, 9]
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for i, axi in enumerate(ax.flat):
        imageId = np.random.randint(0, Xtest.shape[0])
        axi.imshow(Xval[Ytrue != Ypred][ids[i]].reshape(INPUT_WIDTH, INPUT_HEIGHT))    
        axi.set_title("True: {} Pred: {}\nProb: {:.4f}".
                      format(Ytrue[Ytrue != Ypred][ids[i]], Ypred[Ytrue != Ypred][ids[i]], ProbsOfTrue[ids[i]]))
    plt.tight_layout(False)
    print("Top Incorrect Predictions")
    plt.show()
    plt.savefig('incorrect.png')
    
    
#================================================================================
# EVALUATE
#================================================================================

results = tf.nn.softmax(model.predict(Xtest)).numpy()
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,Ntest+1),name = "ImageId"),results],axis = 1)
submission.to_csv("cnn_mnist_datagen.csv",index=False)

if DO_TRAIN:
    print("Training history:")
    print(history.history)
    print("Saving weights")
    model.save_weights(WEIGHTS_NAME)
else:
    print("Model performance on the test set:")
    model.evaluate(Xtest, Ytest, verbose=2)    



