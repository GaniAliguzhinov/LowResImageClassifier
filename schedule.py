from imports import *
from load import load_data

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
def getcallbacks():
    learning_rate_reduction = tf.keras.callbacks.LearningRateScheduler(scheduler)
    callbacks = [learning_rate_reduction]
    return callbacks
