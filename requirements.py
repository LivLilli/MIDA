# import libraries
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import layers, Model, metrics, losses, optimizers, callbacks, models
from tensorflow.keras.models import load_model
import numpy as np
import warnings
warnings.filterwarnings('ignore') #to avoid some ugly warnings
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import figure
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
#tf.random.set_seed(1234)
from tqdm import tqdm
tf.autograph.set_verbosity(1)
import logging
import os
import shutil
# disable logging messages by tf
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.random.set_seed(987)