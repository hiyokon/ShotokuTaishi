#! /usr/bin/env python

import scipy
from scipy import io
from scipy.io import wavfile
from scipy.io.wavfile import read as wavread
# from scikits.talkbox.features import mfcc

import os
import glob
import subprocess
import numpy as np

from matplotlib import pylab
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import ImageGrid # or subplot
from matplotlib.pyplot import specgram
from pylab import subplot, plot, title

from constant import *

def divide_wav(path, sec):
    base, ext = os.path.splitext(path)
    cmd = 'sox %s.wav %s_part_.wav trim 0 %d : newfile : restart' % (base, base, sec)
    subprocess.check_output(cmd.split(" "))
    last = glob.glob(base + "_part_*")[-1]
    os.remove(last)

def create_spectrogram(path):
    output_path = path.replace('.wav', '.png')
    cmd = 'sox %s -n spectrogram -o %s' % (path, output_path)
    subprocess.check_output(cmd.split(" "))

def clear(directory=DATA_DIR):
    for path in glob.glob(os.path.join(directory, '*', '*_part_*')):
        print("Remove: ", path)
        os.remove(path)

def create_wavs(directory=DATA_DIR, sec=3):
    for path in glob.glob(os.path.join(directory, '*', '*.wav')):
      divide_wav(path, sec)

def create_spectrograms(directory=DATA_DIR):
    for path in glob.glob(os.path.join(directory, '*', '*_part_*.wav')):
        create_spectrogram(path)
        print("Create: ", path.replace('wav', '{wav,png}'))

def show_mfccs(numbers, X, y, labels):
  for number, label in zip(numbers, labels):
      subplot(number)
      title(label)
      X_ = [ x for x, y_ in zip(X, y) if y_ == label]
      for x in X_: plot(range(len(x)), x)

def show_spectrogram(path):
    sample_rate, X = wavread(path)
    output = specgram(X, Fs=sample_rate)

def show_spectrograms(directory, label):
    ims = glob.glob(os.path.join(directory, label, '*.png'))
    fig = plt.figure(1, (20., 20.))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, len(ims)), axes_pad=0.1)
    for i, path in enumerate(ims):
        im = mpimg.imread(path)
        grid[i].imshow(im)  # like plt.imshow()
    plt.show()

def show_confusion_matrix(matrix, labels, title=""):
    pylab.clf()
    pylab.matshow(matrix, fignum=False, cmap='Blues', vmin=0, vmax=1.0)
    pylab.title(title)
    pylab.colorbar()
    pylab.grid(False)
    pylab.xlabel('Predicted Class')
    pylab.ylabel('True Class')
    nums = range(len(labels))
    axes = pylab.axes()
    axes.set_xticks(nums)
    axes.set_xticklabels(labels)
    axes.set_yticks(nums)
    axes.set_yticklabels(labels)
    axes.xaxis.set_ticks_position('bottom')
    pylab.show()

