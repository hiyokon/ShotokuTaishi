#! /usr/bin/env python

# =================================
# Modules
# =================================

import os
import glob
import numpy as np
import scipy
from scipy import io
from scipy.io import wavfile
from scipy.io.wavfile import read as wavread
from scikits.talkbox.features import mfcc
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.utils import resample

from constant import *

# =================================
# Global Variables
# =================================

MODEL_LR = LogisticRegression
MODEL_SVM = LinearSVC

# =================================
# Functions
# =================================

def write_ceps(ceps, path):
  output_path = os.path.splitext(path)[0] + ".ceps"
  np.save(output_path, ceps)

def read_as_mfcc(path):
  sample_rate, X = wavread(path)
  ceps, mspec, spec = mfcc(X)
  return ceps

def create_ceps(path):
  sample_rate, X = wavread(path)
  ceps, mspec, spec = mfcc(X)
  write_ceps(ceps, path)

def trim_ceps(ceps, ratio=0.1):
  count = len(ceps)
  start = int(count*ratio)
  end = int(count*(1-ratio))
  return ceps[start:end]

def read_ceps(labels, directory=DATA_DIR):
  X, y = [], []
  for label in labels:
    for path in glob.glob(os.path.join(directory, label, "*.ceps.npy")):
      ceps = np.load(path)
      X.append(np.mean(trim_ceps(ceps), axis=0))
      y.append(label)
  return np.array(X), np.array(y)

def create_ceps_all(directory=DATA_DIR):
  for path in glob.glob(os.path.join(directory, "*", "*_part_*.wav")):
    create_ceps(path)

# def logistic_regression_model(classifier, X):
#   a0 = classifier.intercept_
#   a1 = classifier.coef_
#   denominator = 1 + np.exp(a0+a1*X)
#   return 1 / denominator

def error_count(xs, ys):
  return np.sum([ 1 if x == y else 0 for x, y in zip(xs, ys) ])

def validate(X, y, train, test, model):
    print("TRAIN:", train, "TEST:", test)
    classifier = model()
    classifier.fit(X[train], y[train])
    y_preds = map(classifier.predict, [ x.reshape(1, -1) for x in X[test] ])
    err = error_count(y[test], y_preds)
    cm = confusion_matrix(y[test], y_preds)
    return err, cm

def cross_validate(X, y, n, model):
  kf = KFold(len(X), n_folds=n, shuffle=False)
  return [ validate(X, y, train, test, model) for train, test in kf ]



