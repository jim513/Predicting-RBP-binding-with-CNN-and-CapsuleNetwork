### Avoid warning ###
import warnings
def warn(*args, **kwargs): pass
warnings.warn = warn

### Essential ###
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

bs = 512
N=16
k=20
m=3
l=8

# bs = batch_size
# N = filter number
# k = filter size
# m = pooling size
# l = neuron number of fully connected layer

def set_convolution_layer():
    input_shape =(98+k , 256)
    model = models.Sequential()
    model.add(layers.Conv1D(N, k, padding='valid',input_shape =input_shape))

    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling1D(pool_size=m))
    model.add(layers.Dropout(0.5))

    model.add(layers.Conv1D(N, int(k/2), padding='valid'))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling1D(pool_size=m))
    model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())
    model.add(layers.Dense(l, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(2))
    model.add(layers.Activation('softmax'))

    model.summary()
    return model

def load_train_data():
    ### Load Dataset ###
    D = pd.read_csv("optimumDataset.csv", header=None)

    ### Splitting dataset into X, Y ###
    X_train = D.iloc[:, :-1].values
    Y_train = D.iloc[:, -1].values
    
    ### Random Shuffle ###
    from sklearn.utils import shuffle
    X_train, Y_train = shuffle(X_train, Y_train)  # Avoiding bias
    
    ## Scaling using sci-kit learn ###
    from sklearn.preprocessing import StandardScaler
    scale = StandardScaler()
    X_train = scale.fit_transform(X_train)
    
    print("X_train.shape: {}".format(X_train.shape))
    print("Y_train.shape: {}".format(Y_train.shape))

    return X_train,Y_train

def train_seq_cnn():
    X_train , Y_train = load_train_data()
    model = set_convolution_layer()

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
    model.fit(X_train,Y_train, epochs=5)


#load_train_data()
#set_convolution_layer()
train_seq_cnn()
