
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
import seaborn as sns
from keras.callbacks import EarlyStopping
import kerastuner as kt

TRAIN_FILE = '../input/digit-recognizer/train.csv'
TEST_FILE = '../input/digit-recognizer/test.csv'
FROM_FILE = False
INPUT_WIDTH = 28
INPUT_HEIGHT = 28
CHANNELS = 1
SHOW_PLOTS = True
BS = 64
SCHEDULE = [4, 0, 0]
DO_TRAIN = True
INITIAL_LR = 0.01
DO_TUNE = False
DO_EARLY_STOP = False
WEIGHTS_NAME = 'MODEL_WEIGHTS'
MEAN = 0.5
STD = 0.2
SEED = 42
