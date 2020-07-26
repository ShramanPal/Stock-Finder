import os
import numpy as np
import pandas as pd
from random import randint
from random import seed
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as smapi
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as cns
import datetime as dt
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import plotly.graph_objects as go
from datetime import datetime
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import keras.backend as k
import keras as K
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Activation, Input, Concatenate, Dropout, Conv1D, Flatten, MaxPooling1D, AveragePooling1D, Layer, Lambda
from keras.initializers import glorot_normal
from keras.regularizers import l2
from keras.utils import to_categorical
from sklearn.metrics import f1_score



stocks = {}                                                    #dictionary to store the companies data by thier names
directory = "/home/shraman/Stocks"                             #folder in which the csv files are stored

def file_to_dataframe(filepath, file):                     #func to put data of companies in the dictionary stocks
    file = file[0:-4]                                      #removes the .csv part from the file name while storing in the stocks dictionary
    stocks[file.lower()] = pd.read_csv(filepath)

def file_retrival(folder):                                 #func to find all the csv files in the defined directory
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            path = os.path.join(folder,file)
            file_to_dataframe(path, file)

def preprocess(open, close, high, low, volume):
    open1 = np.array(open[0:-1])
    open2 = np.array(open[1:])
    close1 = np.array(close[0:-1])
    close2 = np.array(close[1:])
    high1 = np.array(close[0:-1])
    high2 = np.array(high[1:])
    low1 = np.array(low[0:-1])
    low2 = np.array(low[1:])
    vol1 = np.array(volume[0:-1])
    vol2 = np.array(volume[1:])

    open_changes = open2 - open1
    close_changes = close2 - close1
    high_changes = high2 - high1
    low_changes = low2 - low1
    volume_changes = vol2 - vol1
    price1 = (open1, close1, high1, low1, vol1)
    price2 = (open2, close2, high2, low2, vol2)

    return open_changes, close_changes, high_changes, low_changes, volume_changes, price1, price2


def series_initialisation(close2, close_changes, open_trend_changes, close_trend_changes, high_trend_changes, low_trend_changes, volume_trend_changes, open_seasonal_changes, close_seasonal_changes, high_seasonal_changes, low_seasonal_changes, volume_seasonal_changes, time_steps, training_size, test_size, neutral_weight):

    f1_train = []
    f2_train = []
    f3_train = []
    f4_train = []
    f5_train = []
    f6_train = []
    f7_train = []
    f8_train = []
    f9_train = []
    f10_train = []
    f1_test = []
    f2_test = []
    f3_test = []
    f4_test = []
    f5_test = []
    f6_test = []
    f7_test = []
    f8_test = []
    f9_test = []
    f10_test = []
    close_train_label = []
    close_test_label = []
    k = -1
    training_indexes = []
    testing_indexes = []
    for j in range(1,training_size + 1):
        k = randint(0,2999)
        while k in training_indexes:
            k = randint(0,2999)
        training_indexes.append(k)
        f1_train.append([[open_trend_changes[i + k]]for i in range(time_steps)])
        f2_train.append([[close_trend_changes[i + k]]for i in range(time_steps)])
        f3_train.append([[low_trend_changes[i + k]]for i in range(time_steps)])
        f4_train.append([[high_trend_changes[i + k]]for i in range(time_steps)])
        f5_train.append([[volume_trend_changes[i + k]]for i in range(time_steps)])
        f6_train.append([[open_seasonal_changes[i + k]]for i in range(time_steps)])
        f7_train.append([[close_seasonal_changes[i + k]]for i in range(time_steps)])
        f8_train.append([[low_seasonal_changes[i + k]]for i in range(time_steps)])
        f9_train.append([[high_seasonal_changes[i + k]]for i in range(time_steps)])
        f10_train.append([[volume_seasonal_changes[i + k]]for i in range(time_steps)])
        if close_changes[k + time_steps]/close2[k + time_steps - 1] >= 0.01:
            close_train_label.append([2])
        elif close_changes[k + time_steps]/close2[k + time_steps - 1] <= -0.01:
            close_train_label.append([0])
        else:
            close_train_label.append([1])

    f1_train = np.array(f1_train)
    f2_train = np.array(f2_train)
    f3_train = np.array(f3_train)
    f4_train = np.array(f4_train)
    f5_train = np.array(f5_train)
    f6_train = np.array(f6_train)
    f7_train = np.array(f7_train)
    f8_train = np.array(f8_train)
    f9_train = np.array(f9_train)
    f10_train = np.array(f10_train)
    close_train_label = np.array(close_train_label)

    print("Open Trend Train Shape: ",f1_train.shape)
    print("Close Trend Train Shape: ",f2_train.shape)
    print("High Trend Train Shape: ",f3_train.shape)
    print("Low Trend Train Shape: ",f4_train.shape)
    print("Volume Trend Train Shape: ",f5_train.shape)
    print("Open Seasonal Train Shape: ",f6_train.shape)
    print("Close Seasonal Train Shape: ",f7_train.shape)
    print("High Seasonal Train Shape: ",f8_train.shape)
    print("Low Seasonal Train Shape: ",f9_train.shape)
    print("Volume Seasonal Train Shape: ",f10_train.shape)

    print("Close Train Label Shape: ",close_train_label.shape)

    for j in range(training_size + 1, training_size + test_size + 1):
        k = randint(0,2999)
        while k in training_indexes:
            k = randint(0,2999)
        training_indexes.append(k)
        f1_test.append([[open_trend_changes[i + k]]for i in range(time_steps)])
        f2_test.append([[close_trend_changes[i + k]]for i in range(time_steps)])
        f3_test.append([[low_trend_changes[i + k]]for i in range(time_steps)])
        f4_test.append([[high_trend_changes[i + k]]for i in range(time_steps)])
        f5_test.append([[volume_trend_changes[i + k]]for i in range(time_steps)])
        f6_test.append([[open_seasonal_changes[i + k]]for i in range(time_steps)])
        f7_test.append([[close_seasonal_changes[i + k]]for i in range(time_steps)])
        f8_test.append([[low_seasonal_changes[i + k]]for i in range(time_steps)])
        f9_test.append([[high_seasonal_changes[i + k]]for i in range(time_steps)])
        f10_test.append([[volume_seasonal_changes[i + k]]for i in range(time_steps)])
        if close_changes[k + time_steps]/close2[k + time_steps] >= 0.01:
            close_test_label.append([2])
        elif close_changes[k + time_steps]/close2[k + time_steps] <= -0.01:
            close_test_label.append([0])
        else:
            close_test_label.append([1])

    f1_test = np.array(f1_test)
    f2_test = np.array(f2_test)
    f3_test = np.array(f3_test)
    f4_test = np.array(f4_test)
    f5_test = np.array(f5_test)
    f6_test = np.array(f6_test)
    f7_test = np.array(f7_test)
    f8_test = np.array(f8_test)
    f9_test = np.array(f9_test)
    f10_test = np.array(f10_test)
    close_test_label = np.array(close_test_label)

    print("Open Trend Test Shape: ",f1_test.shape)
    print("Close Trend Test Shape: ",f2_test.shape)
    print("High Trend Test Shape: ",f3_test.shape)
    print("Low Trend Test Shape: ",f4_test.shape)
    print("Volume Trend Test Shape: ",f5_test.shape)
    print("Open Seasonal Test Shape: ",f6_test.shape)
    print("Close Seasonal Test Shape: ",f7_test.shape)
    print("High Seasonal Test Shape: ",f8_test.shape)
    print("Low Seasonal Test Shape: ",f9_test.shape)
    print("Volume Seasonal Test Shape: ",f10_test.shape)

    print("Close Test Label Shape: ", close_test_label.shape)

    train = (f1_train, f2_train, f3_train, f4_train, f5_train, f6_train, f7_train, f8_train, f9_train, f10_train)
    test = (f1_test, f2_test, f3_test, f4_test, f5_test, f6_test, f7_test, f8_test, f9_test, f10_test)

    close_train_counts = np.unique(close_train_label, return_counts = True)
    close_test_counts = np.unique(close_test_label, return_counts = True)

    print("Train counts: ", close_train_counts)
    print("Test counts: ", close_test_counts)
    class_weights = {0 : training_size/((close_train_counts[1][0] + close_test_counts[1][0]) * 3 ), 1 : training_size/((close_train_counts[1][1] + close_test_counts[1][1])* 3) + neutral_weight, 2 : training_size/((close_train_counts[1][2] + close_test_counts[1][2])* 3)}
    print("Class Weights: ", class_weights)
    close_train_label = to_categorical(close_train_label)
    close_test_label = to_categorical(close_test_label)

    return train, close_train_label, test, close_test_label, class_weights

class attention(Layer):
    def __init__(self, **kwargs):
        super(attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name = "attention_weight", shape = (input_shape[-1], 1), initializer = "normal")
        self.b = self.add_weight(name = "attention bias", shape = (input_shape[1], 1), initializer = "zeros")
        super(attention, self).build(input_shape)

    def call(self, x):
        et = k.squeeze(k.tanh(k.dot(x, self.W) + self.b), axis = -1)
        at = k.softmax(et)
        at = k.expand_dims(at, axis = -1)
        output = x*at
        return k.sum(output, axis = 1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super(attention, self).get_config()


def lstm_model_initialisation(time_steps, lstm_layers , dense_units, regularization_lambda):

    adam = K.optimizers.Adam(learning_rate = 0.0008, beta_1 = 0.9, beta_2 = 0.999)
    initializer = K.initializers.glorot_normal()
    regularizer = l2(regularization_lambda)

    f1_input = Input(shape = (time_steps,1), name = "feature_1")
    f2_input = Input(shape = (time_steps,1), name = "feature_2")
    f3_input = Input(shape = (time_steps,1), name = "feature_3")
    f4_input = Input(shape = (time_steps,1), name = "feature_4")
    f5_input = Input(shape = (time_steps,1), name = "feature_5")
    f6_input = Input(shape = (time_steps,1), name = "feature_6")
    f7_input = Input(shape = (time_steps,1), name = "feature_7")
    f8_input = Input(shape = (time_steps,1), name = "feature_8")
    f9_input = Input(shape = (time_steps,1), name = "feature_9")
    f10_input = Input(shape = (time_steps,1), name = "feature_10")

    f1_layer = LSTM(units = 128, batch_input_shape = (None, time_steps, 1), return_sequences = 1)(f1_input)
    f2_layer = LSTM(units = 128, batch_input_shape = (None, time_steps, 1), return_sequences = 1)(f2_input)
    f3_layer = LSTM(units = 128, batch_input_shape = (None, time_steps, 1), return_sequences = 1)(f3_input)
    f4_layer = LSTM(units = 128, batch_input_shape = (None, time_steps, 1), return_sequences = 1)(f4_input)
    f5_layer = LSTM(units = 128, batch_input_shape = (None, time_steps, 1), return_sequences = 1)(f5_input)
    f6_layer = LSTM(units = 128, batch_input_shape = (None, time_steps, 1), return_sequences = 1)(f6_input)
    f7_layer = LSTM(units = 128, batch_input_shape = (None, time_steps, 1), return_sequences = 1)(f7_input)
    f8_layer = LSTM(units = 128, batch_input_shape = (None, time_steps, 1), return_sequences = 1)(f8_input)
    f9_layer = LSTM(units = 128, batch_input_shape = (None, time_steps, 1), return_sequences = 1)(f9_input)
    f10_layer = LSTM(units = 128, batch_input_shape = (None, time_steps, 1), return_sequences = 1)(f10_input)

    f1_conv = Conv1D(filters = 64, kernel_size = 5, strides = 1, padding = "valid")(f1_layer)
    f2_conv = Conv1D(filters = 64, kernel_size = 5, strides = 1, padding = "valid")(f2_layer)
    f3_conv = Conv1D(filters = 64, kernel_size = 5, strides = 1, padding = "valid")(f3_layer)
    f4_conv = Conv1D(filters = 64, kernel_size = 5, strides = 1, padding = "valid")(f4_layer)
    f5_conv = Conv1D(filters = 64, kernel_size = 5, strides = 1, padding = "valid")(f5_layer)
    f6_conv = Conv1D(filters = 64, kernel_size = 5, strides = 1, padding = "valid")(f6_layer)
    f7_conv = Conv1D(filters = 64, kernel_size = 5, strides = 1, padding = "valid")(f7_layer)
    f8_conv = Conv1D(filters = 64, kernel_size = 5, strides = 1, padding = "valid")(f8_layer)
    f9_conv = Conv1D(filters = 64, kernel_size = 5, strides = 1, padding = "valid")(f9_layer)
    f10_conv = Conv1D(filters = 64, kernel_size = 5, strides = 1, padding = "valid")(f10_layer)


    f1_conv = Conv1D(filters = 32, kernel_size = 3, strides = 1, padding = "valid")(f1_conv)
    f2_conv = Conv1D(filters = 32, kernel_size = 3, strides = 1, padding = "valid")(f2_conv)
    f3_conv = Conv1D(filters = 32, kernel_size = 3, strides = 1, padding = "valid")(f3_conv)
    f4_conv = Conv1D(filters = 32, kernel_size = 3, strides = 1, padding = "valid")(f4_conv)
    f5_conv = Conv1D(filters = 32, kernel_size = 3, strides = 1, padding = "valid")(f5_conv)
    f6_conv = Conv1D(filters = 32, kernel_size = 3, strides = 1, padding = "valid")(f6_conv)
    f7_conv = Conv1D(filters = 32, kernel_size = 3, strides = 1, padding = "valid")(f7_conv)
    f8_conv = Conv1D(filters = 32, kernel_size = 3, strides = 1, padding = "valid")(f8_conv)
    f9_conv = Conv1D(filters = 32, kernel_size = 3, strides = 1, padding = "valid")(f9_conv)
    f10_conv = Conv1D(filters = 32, kernel_size = 3, strides = 1, padding = "valid")(f10_conv)

    #f1_conv = Conv1D(filters = 16, kernel_size = 3, strides = 1, padding = "valid")(f1_conv)
    #f2_conv = Conv1D(filters = 16, kernel_size = 3, strides = 1, padding = "valid")(f2_conv)
    #f3_conv = Conv1D(filters = 16, kernel_size = 3, strides = 1, padding = "valid")(f3_conv)
    #f4_conv = Conv1D(filters = 16, kernel_size = 3, strides = 1, padding = "valid")(f4_conv)
    #f5_conv = Conv1D(filters = 16, kernel_size = 3, strides = 1, padding = "valid")(f5_conv)
    #f6_conv = Conv1D(filters = 16, kernel_size = 3, strides = 1, padding = "valid")(f6_conv)
    #f7_conv = Conv1D(filters = 16, kernel_size = 3, strides = 1, padding = "valid")(f7_conv)
    #f8_conv = Conv1D(filters = 16, kernel_size = 3, strides = 1, padding = "valid")(f8_conv)
    #f9_conv = Conv1D(filters = 16, kernel_size = 3, strides = 1, padding = "valid")(f9_conv)
    #f10_conv = Conv1D(filters = 16, kernel_size = 3, strides = 1, padding = "valid")(f10_conv)

    #f1_conv = Conv1D(filters = 1, kernel_size = 1, strides = 1, padding = "valid")(f1_conv)
    #f2_conv = Conv1D(filters = 1, kernel_size = 1, strides = 1, padding = "valid")(f2_conv)
    #f3_conv = Conv1D(filters = 1, kernel_size = 1, strides = 1, padding = "valid")(f3_conv)
    #f4_conv = Conv1D(filters = 1, kernel_size = 1, strides = 1, padding = "valid")(f4_conv)
    #f5_conv = Conv1D(filters = 1, kernel_size = 1, strides = 1, padding = "valid")(f5_conv)
    #f6_conv = Conv1D(filters = 1, kernel_size = 1, strides = 1, padding = "valid")(f6_conv)
    #f7_conv = Conv1D(filters = 1, kernel_size = 1, strides = 1, padding = "valid")(f7_conv)
    #f8_conv = Conv1D(filters = 1, kernel_size = 1, strides = 1, padding = "valid")(f8_conv)
    #f9_conv = Conv1D(filters = 1, kernel_size = 1, strides = 1, padding = "valid")(f9_conv)
    #f10_conv = Conv1D(filters = 1, kernel_size = 1, strides = 1, padding = "valid")(f10_conv)

    #f1_flatten = Flatten()(f1_conv)
    #f2_flatten = Flatten()(f2_conv)
    #f3_flatten = Flatten()(f3_conv)
    #f4_flatten = Flatten()(f4_conv)
    #f5_flatten = Flatten()(f5_conv)
    #f6_flatten = Flatten()(f6_conv)
    #f7_flatten = Flatten()(f7_conv)
    #f8_flatten = Flatten()(f8_conv)
    #f9_flatten = Flatten()(f9_conv)
    #f10_flatten = Flatten()(f10_conv)

    f1_flatten = (f1_conv)
    f2_flatten = (f2_conv)
    f3_flatten = (f3_conv)
    f4_flatten = (f4_conv)
    f5_flatten = (f5_conv)
    f6_flatten = (f6_conv)
    f7_flatten = (f7_conv)
    f8_flatten = (f8_conv)
    f9_flatten = (f9_conv)
    f10_flatten = (f10_conv)



    #f1_flatten = Dense(units = 8, activation = "tanh", kernel_initializer = initializer, kernel_regularizer = regularizer)(f1_flatten)
    #f2_flatten = Dense(units = 8, activation = "tanh", kernel_initializer = initializer, kernel_regularizer = regularizer)(f2_flatten)
    #f3_flatten = Dense(units = 8, activation = "tanh", kernel_initializer = initializer, kernel_regularizer = regularizer)(f3_flatten)
    #f4_flatten = Dense(units = 8, activation = "tanh", kernel_initializer = initializer, kernel_regularizer = regularizer)(f4_flatten)
    #f5_flatten = Dense(units = 8, activation = "tanh", kernel_initializer = initializer, kernel_regularizer = regularizer)(f5_flatten)
    #f6_flatten = Dense(units = 8, activation = "tanh", kernel_initializer = initializer, kernel_regularizer = regularizer)(f6_flatten)
    #f7_flatten = Dense(units = 8, activation = "tanh", kernel_initializer = initializer, kernel_regularizer = regularizer)(f7_flatten)
    #f8_flatten = Dense(units = 8, activation = "tanh", kernel_initializer = initializer, kernel_regularizer = regularizer)(f8_flatten)
    #f9_flatten = Dense(units = 8, activation = "tanh", kernel_initializer = initializer, kernel_regularizer = regularizer)(f9_flatten)
    #f10_flatten = Dense(units = 8, activation = "tanh", kernel_initializer = initializer, kernel_regularizer = regularizer)(f10_flatten)

    f1_flatten = attention()(f1_flatten)
    f2_flatten = attention()(f2_flatten)
    f3_flatten = attention()(f3_flatten)
    f4_flatten = attention()(f4_flatten)
    f5_flatten = attention()(f5_flatten)
    f6_flatten = attention()(f6_flatten)
    f7_flatten = attention()(f7_flatten)
    f8_flatten = attention()(f8_flatten)
    f9_flatten = attention()(f9_flatten)
    f10_flatten = attention()(f10_flatten)

    output = Concatenate()([f1_flatten, f2_flatten, f3_flatten, f4_flatten, f6_flatten, f7_flatten, f8_flatten, f9_flatten])
    for i,d in enumerate(dense_units):
        output = (Dense(d, activation = "tanh", kernel_initializer = initializer, kernel_regularizer = regularizer))(output)
        if i%3 == 0:
            output = Dropout(rate = 0.15)(output)
    output = Dense(8, activation = "tanh", kernel_initializer = initializer)(output)
    output = Dense(3, activation = "softmax", kernel_initializer = initializer)(output)

    model = Model(inputs = [f1_input, f2_input, f3_input, f4_input, f5_input, f6_input, f7_input, f8_input, f9_input, f10_input], outputs = output)
    model.compile(optimizer = adam, loss = "categorical_crossentropy", metrics = [K.metrics.Precision(class_id = 0), K.metrics.Precision(class_id = 1), K.metrics.Precision(class_id = 2), K.metrics.Recall(class_id = 0), K.metrics.Recall(class_id = 1), K.metrics.Recall(class_id = 2)])

    print(model.summary())
    return model

def cnn_model_initialisation(time_steps , conv_units, regularization_lambda):

    adam = K.optimizers.Adam(learning_rate = 0.0003, beta_1 = 0.9, beta_2 = 0.999)
    initializer = K.initializers.glorot_normal()
    regularizer = l2(regularization_lambda)

    f1_input = Input(shape = (time_steps,1), name = "feature_1")
    f2_input = Input(shape = (time_steps,1), name = "feature_2")
    f3_input = Input(shape = (time_steps,1), name = "feature_3")
    f4_input = Input(shape = (time_steps,1), name = "feature_4")
    f5_input = Input(shape = (time_steps,1), name = "feature_5")

    f1_conv = Conv1D(filters = 128,kernel_size = 5,strides = 1,padding = "valid",kernel_initializer = initializer, kernel_regularizer = regularizer)(f1_input)
    f2_conv = Conv1D(filters = 128,kernel_size = 5,strides = 1,padding = "valid",kernel_initializer = initializer, kernel_regularizer = regularizer)(f2_input)
    f3_conv = Conv1D(filters = 128,kernel_size = 5,strides = 1,padding = "valid",kernel_initializer = initializer, kernel_regularizer = regularizer)(f3_input)
    f4_conv = Conv1D(filters = 128,kernel_size = 5,strides = 1,padding = "valid",kernel_initializer = initializer, kernel_regularizer = regularizer)(f4_input)
    f5_conv = Conv1D(filters = 128,kernel_size = 5,strides = 1,padding = "valid",kernel_initializer = initializer, kernel_regularizer = regularizer)(f5_input)

    for i,f in enumerate(conv_units):
        if i < 2:
            f1_conv = Conv1D(filters = f, kernel_size = 7, strides = 1, padding = "valid", kernel_initializer = initializer, kernel_regularizer = regularizer)(f1_conv)
            f2_conv = Conv1D(filters = f, kernel_size = 7, strides = 1, padding = "valid", kernel_initializer = initializer, kernel_regularizer = regularizer)(f2_conv)
            f3_conv = Conv1D(filters = f, kernel_size = 7, strides = 1, padding = "valid", kernel_initializer = initializer, kernel_regularizer = regularizer)(f3_conv)
            f4_conv = Conv1D(filters = f, kernel_size = 7, strides = 1, padding = "valid", kernel_initializer = initializer, kernel_regularizer = regularizer)(f4_conv)
            f5_conv = Conv1D(filters = f, kernel_size = 7, strides = 1, padding = "valid", kernel_initializer = initializer, kernel_regularizer = regularizer)(f5_conv)
            #f1_conv = MaxPooling1D(pool_size = 3, strides = 1, padding = "valid")(f1_conv)
            #f2_conv = MaxPooling1D(pool_size = 3, strides = 1, padding = "valid")(f2_conv)
            #f3_conv = MaxPooling1D(pool_size = 3, strides = 1, padding = "valid")(f3_conv)
            #f4_conv = MaxPooling1D(pool_size = 3, strides = 1, padding = "valid")(f4_conv)
            #f5_conv = MaxPooling1D(pool_size = 3, strides = 1, padding = "valid")(f5_conv)
        else:
            f1_conv = Conv1D(filters = f, kernel_size = 7, strides = 1, padding = "valid", kernel_initializer = initializer, kernel_regularizer = regularizer)(f1_conv)
            f2_conv = Conv1D(filters = f, kernel_size = 7, strides = 1, padding = "valid", kernel_initializer = initializer, kernel_regularizer = regularizer)(f2_conv)
            f3_conv = Conv1D(filters = f, kernel_size = 7, strides = 1, padding = "valid", kernel_initializer = initializer, kernel_regularizer = regularizer)(f3_conv)
            f4_conv = Conv1D(filters = f, kernel_size = 7, strides = 1, padding = "valid", kernel_initializer = initializer, kernel_regularizer = regularizer)(f4_conv)
            f5_conv = Conv1D(filters = f, kernel_size = 7, strides = 1, padding = "valid", kernel_initializer = initializer, kernel_regularizer = regularizer)(f5_conv)


    f1_flatten = Flatten()(f1_conv)
    f2_flatten = Flatten()(f2_conv)
    f3_flatten = Flatten()(f3_conv)
    f4_flatten = Flatten()(f4_conv)
    f5_flatten = Flatten()(f5_conv)

    f1_dense = Dense(units = 32, activation = "tanh", kernel_initializer = initializer, kernel_regularizer = regularizer)(f1_flatten)
    f2_dense = Dense(units = 32, activation = "tanh",kernel_initializer = initializer, kernel_regularizer = regularizer)(f2_flatten)
    f3_dense = Dense(units = 32, activation = "tanh", kernel_initializer = initializer, kernel_regularizer = regularizer)(f3_flatten)
    f4_dense = Dense(units = 32, activation = "tanh", kernel_initializer = initializer, kernel_regularizer = regularizer)(f4_flatten)
    f5_dense = Dense(units = 32, activation = "tanh", kernel_initializer = initializer, kernel_regularizer = regularizer)(f5_flatten)

    output = Concatenate()([f1_dense, f2_dense, f3_dense, f4_dense, f5_dense])
    output = Dense(units = 64, activation = "tanh", kernel_initializer = initializer, kernel_regularizer = regularizer)(output)
    output = Dense(units = 32, activation = "tanh", kernel_initializer = initializer, kernel_regularizer = regularizer)(output)
    output = Dense(units = 8, activation = "tanh", kernel_initializer = initializer, kernel_regularizer = regularizer)(output)
    output = Dense(units = 3, activation = "softmax", kernel_initializer = initializer, kernel_regularizer = regularizer)(output)

    model = Model(inputs = [f1_input, f2_input, f3_input, f4_input, f5_input], outputs = output)
    model.compile(loss = "categorical_crossentropy", optimizer = adam, metrics = [K.metrics.Precision(class_id = 0), K.metrics.Precision(class_id = 1), K.metrics.Precision(class_id = 2), K.metrics.Recall(class_id = 0), K.metrics.Recall(class_id = 1), K.metrics.Recall(class_id = 2)])
    print(model.summary())
    return model

def f1score(model, test, test_label):
    epsilon = 0.000001
    open_trend_test, close_trend_test, low_trend_test, high_trend_test, vol_trend_test, open_seasonal_test, close_seasonal_test, low_seasonal_test, high_seasonal_test, vol_seasonal_test = test
    score = model.evaluate([open_trend_test, close_trend_test, low_trend_test, high_trend_test, vol_trend_test, open_seasonal_test, close_seasonal_test, low_seasonal_test, high_seasonal_test, vol_seasonal_test], test_label)

    print("Individual Scores: ", score)
    print("F1 Score of Down: ", (2*score[1]*score[4])/(score[1] + score[4] + epsilon))
    print("F1 Score of Up: ", (2*score[3]*score[6])/(score[3] + score[6] + epsilon))
    print("F1 Score of Neutral: ", (2*score[2]*score[5])/(score[2] + score[5] + epsilon))

def lstm_loss_plot(history):
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.show()
    plt.plot(history.history["precision_1"])
    plt.plot(history.history["recall_1"])
    plt.show()
    plt.plot(history.history["precision_2"])
    plt.plot(history.history["recall_2"])
    plt.show()
    plt.plot(history.history["precision_3"])
    plt.plot( history.history["recall_3"])
    plt.show()
    plt.plot(history.history["val_precision_1"])
    plt.plot(history.history["val_recall_1"])
    plt.show()
    print("LSTM Val class 1: ", 2*history.history["val_precision_1"][-1]*history.history["val_recall_1"][-1]/(history.history["val_precision_1"][-1] + history.history["val_recall_1"][-1] + 0.000001))
    plt.plot(history.history["val_precision_2"])
    plt.plot(history.history["val_recall_2"])
    plt.show()
    print("LSTM Val class 2: ", 2*history.history["val_precision_2"][-1]*history.history["val_recall_2"][-1]/(history.history["val_precision_2"][-1] + history.history["val_recall_2"][-1] + 0.000001))
    plt.plot(history.history["val_precision_3"])
    plt.plot(history.history["val_recall_3"])
    plt.show()
    print("LSTM Val class 3: ", 2*history.history["val_precision_3"][-1]*history.history["val_recall_3"][-1]/(history.history["val_precision_3"][-1] + history.history["val_recall_3"][-1] + 0.000001))

def cnn_loss_plot(history):
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.show()
    plt.plot(history.history["precision_4"])
    plt.plot(history.history["recall_4"])
    plt.show()
    plt.plot(history.history["precision_5"])
    plt.plot(history.history["recall_5"])
    plt.show()
    plt.plot(history.history["precision_6"])
    plt.plot( history.history["recall_6"])
    plt.show()
    plt.plot(history.history["val_precision_4"])
    plt.plot(history.history["val_recall_4"])
    plt.show()
    print("CNN Val class 1: ", 2*history.history["val_precision_4"][-1]*history.history["val_recall_4"][-1]/(history.history["val_precision_4"][-1] + history.history["val_recall_4"][-1] + 0.000001))
    plt.plot(history.history["val_precision_5"])
    plt.plot(history.history["val_recall_5"])
    plt.show()
    print("CNN Val class 2: ", 2*history.history["val_precision_5"][-1]*history.history["val_recall_5"][-1]/(history.history["val_precision_5"][-1] + history.history["val_recall_5"][-1] + 0.000001))
    plt.plot(history.history["val_precision_6"])
    plt.plot(history.history["val_recall_6"])
    plt.show()
    print("CNN Val class 3: ", 2*history.history["val_precision_6"][-1]*history.history["val_recall_6"][-1]/(history.history["val_precision_6"][-1] + history.history["val_recall_6"][-1] + 0.000001))

def attention_loss_plot(history):
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.show()
    plt.plot(history.history["precision_7"])
    plt.plot(history.history["recall_7"])
    plt.show()
    plt.plot(history.history["precision_8"])
    plt.plot(history.history["recall_8"])
    plt.show()
    plt.plot(history.history["precision_9"])
    plt.plot( history.history["recall_9"])
    plt.show()
    plt.plot(history.history["val_precision_7"])
    plt.plot(history.history["val_recall_7"])
    plt.show()
    print("Attention Val class 1: ", 2*history.history["val_precision_7"][-1]*history.history["val_recall_7"][-1]/(history.history["val_precision_7"][-1] + history.history["val_recall_7"][-1] + 0.000001))
    plt.plot(history.history["val_precision_8"])
    plt.plot(history.history["val_recall_8"])
    plt.show()
    print("Attention Val class 2: ", 2*history.history["val_precision_8"][-1]*history.history["val_recall_8"][-1]/(history.history["val_precision_8"][-1] + history.history["val_recall_8"][-1] + 0.000001))
    plt.plot(history.history["val_precision_9"])
    plt.plot(history.history["val_recall_9"])
    plt.show()
    print("Attention Val class 3: ", 2*history.history["val_precision_9"][-1]*history.history["val_recall_9"][-1]/(history.history["val_precision_9"][-1] + history.history["val_recall_9"][-1] + 0.000001))

def attention_model_initialisation(time_steps, regularization_lambda):
    adam = K.optimizers.Adam(learning_rate = 0.0008, beta_1 = 0.9, beta_2 = 0.999)
    initializer = K.initializers.glorot_normal()
    regularizer = l2(regularization_lambda)

    f1_input = Input(shape = (time_steps,1), name = "feature_1")
    f2_input = Input(shape = (time_steps,1), name = "feature_2")
    f3_input = Input(shape = (time_steps,1), name = "feature_3")
    f4_input = Input(shape = (time_steps,1), name = "feature_4")
    f5_input = Input(shape = (time_steps,1), name = "feature_5")

    f1_layer = LSTM(units = 256, batch_input_shape = (None, time_steps, 1), return_sequences = 1)(f1_input)
    f2_layer = LSTM(units = 256, batch_input_shape = (None, time_steps, 1), return_sequences = 1)(f2_input)
    f3_layer = LSTM(units = 256, batch_input_shape = (None, time_steps, 1), return_sequences = 1)(f3_input)
    f4_layer = LSTM(units = 256, batch_input_shape = (None, time_steps, 1), return_sequences = 1)(f4_input)
    f5_layer = LSTM(units = 256, batch_input_shape = (None, time_steps, 1), return_sequences = 1)(f5_input)

    f1_attention = attention()(f1_layer)
    f2_attention = attention()(f2_layer)
    f3_attention = attention()(f3_layer)
    f4_attention = attention()(f4_layer)
    f5_attention = attention()(f5_layer)

    f1_dense = Dense(units = 32, activation = "relu", kernel_initializer = initializer, kernel_regularizer = regularizer)(f1_attention)
    f2_dense = Dense(units = 32, activation = "relu", kernel_initializer = initializer, kernel_regularizer = regularizer)(f2_attention)
    f3_dense = Dense(units = 32, activation = "relu", kernel_initializer = initializer, kernel_regularizer = regularizer)(f3_attention)
    f4_dense = Dense(units = 32, activation = "relu", kernel_initializer = initializer, kernel_regularizer = regularizer)(f4_attention)
    f5_dense = Dense(units = 32, activation = "relu", kernel_initializer = initializer, kernel_regularizer = regularizer)(f5_attention)

    output = Concatenate()([f1_dense, f2_dense, f3_dense, f4_dense, f5_dense])

    output = Dense(units = 128, activation = "tanh", kernel_initializer = initializer, kernel_regularizer = regularizer)(output)
    output = Dense(units = 64, activation = "tanh", kernel_initializer = initializer, kernel_regularizer = regularizer)(output)
    output = Dense(units = 32, activation = "tanh", kernel_initializer = initializer, kernel_regularizer = regularizer)(output)
    output = Dense(units = 16, activation = "tanh", kernel_initializer = initializer, kernel_regularizer = regularizer)(output)
    output = Dense(units = 8, activation = "tanh", kernel_initializer = initializer, kernel_regularizer = regularizer)(output)
    output = Dense(units = 3, activation = "softmax", kernel_initializer = initializer, kernel_regularizer = regularizer)(output)

    model = Model(inputs = [f1_input, f2_input, f3_input, f4_input, f5_input], outputs = output)
    model.compile(loss = "categorical_crossentropy", optimizer = adam, metrics = [K.metrics.Precision(class_id = 0), K.metrics.Precision(class_id = 1), K.metrics.Precision(class_id = 2), K.metrics.Recall(class_id = 0), K.metrics.Recall(class_id = 1), K.metrics.Recall(class_id = 2)])
    print(model.summary())
    return model

def mape_prediction(model, xtest, ytest):
    output = model.predict(xtest)
    print("Shape of output: ", output.shape)
    print("Numer of test examples: ", len(ytest))
    mape = 0
    for i,s in enumerate(output):
        print(i,": ", output[i]*sd+avg, " ", ytest[i]*sd+avg)
        mape += np.sum(abs((output[i] - ytest[i])/ytest[i]))
    print(mape/len(ytest))

if __name__ == "__main__":
    file_retrival(directory)                                 #extracting the csv files into the dictionary stocks
    stocks.pop("stock_metadata")                             #removing the unrequired extra csv file
    for x in stocks.keys():
        stock = stocks[x]
    adani = stocks["adaniports"]
    open = adani["Open"]
    close = adani["Close"]
    high = adani["High"]
    low = adani["Low"]
    volumes = adani["Volume"]
    volumes = np.array(volumes[:])
    volumes = (volumes - volumes.mean())/volumes.std()
    lstm_layers = 1
    dense_units = [128, 32, 16]
    cnn_units = [128, 128, 128, 32, 8]
    time_steps = 90
    training_size = 2500
    test_size = 420
    neutral_weight = 0
    regularization_lambda_lstm = 0.007
    regularization_lambda_cnn = 0.0001
    regularization_lambda_attention = 0.0

    open_seasonal, open_trend = smapi.tsa.filters.hpfilter(open, 100)
    close_seasonal, close_trend = smapi.tsa.filters.hpfilter(close, 100)
    high_seasonal, high_trend = smapi.tsa.filters.hpfilter(high, 100)
    low_seasonal, low_trend = smapi.tsa.filters.hpfilter(low, 100)
    volume_seasonal, volume_trend = smapi.tsa.filters.hpfilter(volumes, 100)

    open_changes, close_changes, high_changes, low_changes, volume_changes, price1, price2 = preprocess(open, close, high, low, volumes)
    open_trend_changes, close_trend_changes, high_trend_changes, low_trend_changes, volume_trend_changes, price1_trend, price2_trend = preprocess(open_trend, close_trend, high_trend, low_trend, volume_trend)
    open_seasonal_changes, close_seasonal_changes, high_seasonal_changes, low_seasonal_changes, volume_seasonal_changes, price1_seasonal, price2_seasonal = preprocess(open_seasonal, close_seasonal, high_seasonal, low_seasonal, volume_seasonal)

    open2, close2, high2, low2, vol2 = price2
    #train, train_label, test, test_label, class_weight = series_initialisation(close2, close_changes, open_trend_changes, close_trend_changes, high_trend_changes, low_trend_changes, volume_trend_changes, open_seasonal_changes, close_seasonal_changes, high_seasonal_changes, low_seasonal_changes, volume_seasonal_changes, time_steps, training_size, test_size, neutral_weight)
    train, train_label, test, test_label, class_weight = series_initialisation(close2, close_changes, open_trend_changes, close_trend_changes, high_trend_changes, low_trend_changes, volume_trend_changes, open_seasonal_changes, close_seasonal_changes, high_seasonal_changes, low_seasonal_changes, volume_seasonal_changes, time_steps, training_size, test_size, neutral_weight)
    open_trend_train, close_trend_train, low_trend_train, high_trend_train, volume_trend_train, open_seasonal_train, close_seasonal_train, low_seasonal_train, high_seasonal_train, volume_seasonal_train = train
    open_trend_test, close_trend_test, low_trend_test, high_test_trend, volume_trend_test, open_seasonal_test, close_seasonal_test, low_seasonal_test, high_seasonal_test, volume_seasonal_test = test

    '''


    for i in range(2989):
        plt.plot(open[i:i + 100], close[i:i + 100], c ="r")
        plt.show()
        plt.plot(high[i:i + 100], close[i:i + 100], c = "b")
        plt.show()
        plt.plot(low[i:i + 100], close[i:i + 100], c = "g")
        plt.show()

    '''
    lstm_model = lstm_model_initialisation(time_steps, lstm_layers, dense_units, regularization_lambda_lstm)
    lstm_history = lstm_model.fit([open_trend_train, close_trend_train, high_trend_train, low_trend_train, volume_trend_train, open_seasonal_train, close_seasonal_train, high_seasonal_train, low_seasonal_train, volume_seasonal_train], train_label, epochs = 350, verbose = 1, batch_size = 1000, validation_split = 0.2, class_weight = class_weight)
    #cnn_model = cnn_model_initialisation(time_steps, cnn_units, regularization_lambda_cnn)
    #cnn_history = cnn_model.fit([open_train, close_train, high_train, low_train, volume_train], train_label, epochs = 150, verbose = 1, batch_size = 500, validation_split = 0.2, class_weight = class_weight)
    #attention_model = attention_model_initialisation(time_steps, regularization_lambda_attention)
    #attention_history = attention_model.fit([open_train, close_train, high_train, low_train, volume_train], train_label, epochs = 200, verbose = 1, batch_size = 500, validation_split = 0.2, class_weight = class_weight)
    lstm_loss_plot(lstm_history)
    #cnn_loss_plot(cnn_history)
    #attention_loss_plot(attention_history)
    f1score(lstm_model, test, test_label)
    #f1score(cnn_model, test, test_label)
    #f1score(attention_model, test, test_label)
    #mape_prediction(model, xtest, ytest)
