import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras.losses import BinaryCrossentropy, MeanAbsoluteError, MeanSquaredError
from tensorflow.keras import backend as K
import os

"""
This file is meant for:
1. callbacks
2. losses
3. metrics

"""

def find_last_checkpoint(checkpoint_path):
    """
    :param checkpoint_path: path to directory where checkpoints should be.
    :return: path to latest checkpoint found in directory,
    """
    files = [os.path.join(checkpoint_path, file) for file in os.listdir(checkpoint_path) if file.endswith('.hdf5')]
    if files:
        return max(files, key=os.path.getctime)
    return None



def train_callbacks(log_path, checkpoint_path):
    """
    Prepares callbacks for training.
    :param log_path: directory to save training log
    :param checkpoint_path: directory to save model checkpoints.
    :return: list of callbacks.
    """
    callbacks = []
    callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                       factor=0.1,
                                                       patience=5,
                                                       verbose=1,
                                                       mode='auto',
                                                       cooldown=0,
                                                       min_lr=1e-7))
    callbacks.append(keras.callbacks.CSVLogger(os.path.join(log_path, 'training_log.csv'), append=True))
    callbacks.append(keras.callbacks.ModelCheckpoint(
        os.path.join(log_path, "saved-model-{epoch:04d}-{val_loss:.4f}.hdf5"),
        monitor='val_loss',
        verbose=1,
        save_best_only=False,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch'))
    return callbacks



# Losses:
def decoding_L1_loss(y_true, y_pred):
    return MeanAbsoluteError()(y_true, y_pred)


def decoding_L2_loss(y_true, y_pred):
    return MeanSquaredError()(y_true, y_pred)


def encoding_deviation_loss(y_true, y_pred):
    deviations = tf.where(y_true < 0, tf.abs(y_true + 1), tf.abs(y_true - 1))
    return tf.reduce_mean(deviations)

# def decoding_bec_loss(y_true, y_pred):
#     #  hard decision:
#     y_hard_decision = 0.5*(tf.ones_like(y_pred) - tf.sign(y_pred))



# Metrics:
# Assumes a single sample! So _y_true and y_pred are single tensor each of shape=(batch_sz, out_len)
def bit_error_rate(y_true, y_pred):
    """
    :return: batch BER
    """
    true_hard_decided = tf.sign(y_true)
    pred_hard_decided = tf.sign(y_pred)
    diff = 0.5 * (true_hard_decided - pred_hard_decided)
    diff = tf.reshape(diff, [-1])
    return tf.reduce_mean(tf.abs(diff))


def median_encoding_deviation(y_true, y_pred):
    """
    :return: Mean of how far encoded bits are from +-1.
    """
    deviations = tf.where(y_true < 0, tf.abs(y_true + 1), tf.abs(y_true - 1))
    return tfp.stats.percentile(tf.reshape(deviations, [-1]), 50.0)


def max_encoding_deviation(y_true, y_pred):
    deviations = tf.where(y_true < 0, tf.abs(y_true + 1), tf.abs(y_true - 1))
    return tf.reduce_max(deviations)





