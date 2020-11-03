from imports import *
from tensorflow.keras import backend as K
import tensorflow as tf
import math
from load import load_data
def _1cycle_mom(iteration_idx:int,
                cyc_iterations:int,
                min_mom:float,
                max_mom:float):
    'TODO: docstring'
    mid = math.floor((cyc_iterations - 1)/2)
    if iteration_idx == mid: return min_mom
    elif iteration_idx == 0 or iteration_idx >= (2 * mid): return max_mom
    else:
        mod = (iteration_idx % mid)
        numerator =  mod if iteration_idx < mid else mid - mod
        return max_mom - (numerator / mid) * (max_mom - min_mom)

def _1cycle_lr(iteration_idx:int,
              cyc_iterations:int,
              ramp_iterations:int,
              min_lr:float,
              max_lr:float):
    'TODO: docstring'
    mid = math.floor((cyc_iterations - 1)/2)
    if iteration_idx == mid: return max_lr
    elif iteration_idx == 0 or iteration_idx == (2 * mid): return min_lr
    elif iteration_idx < cyc_iterations:
        mod = (iteration_idx % mid)
        numerator =  mod if iteration_idx < mid else mid - mod
        return min_lr + (numerator / mid) * (max_lr - min_lr)
    else:
        idx = iteration_idx - cyc_iterations
        ramp_max = min_lr
        ramp_min = min_lr * 1e-5
        return ramp_max - ((idx + 1) / ramp_iterations) * (ramp_max - ramp_min)

def _inv_lr(iteration_idx:int, gamma:float, power:float,  base_lr:float):
    'ported from: https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto#L157-L172'
    return base_lr * (1 + gamma * iteration_idx) ** (- power)
class OneCycleSchedulerCallback(tf.keras.callbacks.Callback):

    def __init__(self,
                 cyc_iterations:int,
                 ramp_iterations:int,
                 min_lr:float,
                 max_lr:float,
                 min_mom:float,
                 max_mom:float):
        'TODO: docstring'
        self.cyc_iterations = cyc_iterations
        self.ramp_iterations = ramp_iterations
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.min_mom = min_mom
        self.max_mom = max_mom
        self.iteration = 0

    def on_batch_begin(self, batch, logs=None):
        'TODO: docstring'
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        if not hasattr(self.model.optimizer, 'momentum'):
            raise ValueError('Optimizer must have a "momentum" attribute.')
        lr = _1cycle_lr(self.iteration, self.cyc_iterations, self.ramp_iterations, self.min_lr, self.max_lr)
        mom = _1cycle_mom(self.iteration, self.cyc_iterations, self.min_mom, self.max_mom)
        K.set_value(self.model.optimizer.lr, lr)
        K.set_value(self.model.optimizer.momentum, mom)

    def on_batch_end(self, batch, logs=None):
        'TODO: docstring'
        self.iteration +=1

    @staticmethod
    def plot_schedule(cyc_iterations:int,
                      ramp_iterations:int,
                      min_lr:float,
                      max_lr:float,
                      min_mom:float,
                      max_mom:float):
        xs = range(cyc_iterations + ramp_iterations)
        lr_ys = [_1cycle_lr(i, cyc_iterations, ramp_iterations, min_lr, max_lr) for i in xs]
        mom_ys = [_1cycle_mom(i, cyc_iterations, min_lr, max_lr) for i in xs]
        lr_df = pd.DataFrame({'iteration': xs, 'lr' : lr_ys})
        mom_df = pd.DataFrame({'iteration': xs, 'mom' : mom_ys})
        _ ,(ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 20))
        sns.lineplot(x='iteration', y='lr', data= lr_df, ax=ax1)
        sns.lineplot(x='iteration', y='mom', data= mom_df, ax=ax2)
def scheduler(epoch, lr, INIT_LR=INITIAL_LR, schedule=SCHEDULE):
    if epoch < schedule[0]:
        return INIT_LR
    elif epoch < schedule[0] + schedule[1]*1:
        return INIT_LR * 0.5
    elif epoch < schedule[0] + schedule[1]*2:
        return INIT_LR * 0.05
    elif epoch < schedule[0] + schedule[1]*2 + schedule[2]:
        return INIT_LR * 0.005
    elif epoch < schedule[0] + schedule[1]*2 + schedule[2]*2:
        return INIT_LR * 0.001
    elif epoch < schedule[0] + schedule[1]*2 + schedule[2]*3:
        return INIT_LR * 0.0001
    else:
        return INIT_LR * 0.001
def totalepochs(schedule):
    return schedule[0] + schedule[1]*2 + schedule[2]*3

def getcallbacks(Xtrain):
    learning_rate_reduction = tf.keras.callbacks.LearningRateScheduler(scheduler)
    callbacks = [learning_rate_reduction]
    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=1, mode='min', min_delta=0.0001, cooldown=0, min_lr=0.0001)]
    cb = OneCycleSchedulerCallback(cyc_iterations=2*5*math.ceil(Xtrain.shape[0] / BS) + 1,
                                   ramp_iterations=2*math.ceil(Xtrain.shape[0] / BS),
                                   min_lr=INITIAL_LR,
                                   max_lr=INITIAL_LR*10,
                                   min_mom=0.8,
                                   max_mom=0.95)
    callbacks = [cb]
    return callbacks
