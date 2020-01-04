


# 利用sklearn自建评价函数
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback

import keras.backend as K
from keras.metrics import binary_accuracy

import tensorflow as tf
import numpy as np

class RocAucEvaluation(Callback):
    def __init__(self, validation_generator, interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        
        validation_data = next(validation_generator)
        
        self.x_val,self.y_val = validation_data
    def on_epoch_end(self, epoch, log={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.x_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print('\n ROC_AUC - epoch:%d - score:%.6f \n' % (epoch+1, score))

'''
x_train,y_train,x_label,y_label = train_test_split(train_feature, train_label, train_size=0.95, random_state=233)
RocAuc = RocAucEvaluation(validation_data=(y_train,y_label), interval=1)


hist = model.fit(x_train, x_label, batch_size=batch_size, epochs=epochs, validation_data=(y_train, y_label), callbacks=[RocAuc], verbose=2)
'''

def auc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

'''
def auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)
'''
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P



def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.
    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    F1 score: https://en.wikipedia.org/wiki/F1_score
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1)


def invasion_acc(y_true, y_pred):
    binary_truth = y_true[:, -2] + y_true[:, -1]
    binary_pred = y_pred[:, -2] + y_pred[:, -1]
    return binary_accuracy(binary_truth, binary_pred)


def invasion_precision(y_true, y_pred):
    binary_truth = y_true[:, -2] + y_true[:, -1]
    binary_pred = y_pred[:, -2] + y_pred[:, -1]
    return precision(binary_truth, binary_pred)


def invasion_recall(y_true, y_pred):
    binary_truth = y_true[:, -2] + y_true[:, -1]
    binary_pred = y_pred[:, -2] + y_pred[:, -1]
    return recall(binary_truth, binary_pred)


def invasion_fmeasure(y_true, y_pred):
    binary_truth = y_true[:, -2] + y_true[:, -1]
    binary_pred = y_pred[:, -2] + y_pred[:, -1]
    return fmeasure(binary_truth, binary_pred)


def ia_acc(y_true, y_pred):
    binary_truth = y_true[:, -1]
    binary_pred = y_pred[:, -1]
    return binary_accuracy(binary_truth, binary_pred)


def ia_precision(y_true, y_pred):
    binary_truth = y_true[:, -1]
    binary_pred = y_pred[:, -1]
    return precision(binary_truth, binary_pred)


def ia_recall(y_true, y_pred):
    binary_truth = y_true[:, -1]
    binary_pred = y_pred[:, -1]
    return recall(binary_truth, binary_pred)


def ia_fmeasure(y_true, y_pred):
    binary_truth = y_true[:, -1]
    binary_pred = y_pred[:, -1]
    return fmeasure(binary_truth, binary_pred)