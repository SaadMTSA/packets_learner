import numpy as np
import keras

from sklearn.linear_model import LogisticRegression
from .base import BaseModel
from skopt import BayesSearchCV
from keras.models import Sequential
from keras.layers import Dense, Activation

class RNNModel(BaseModel):
    def __init__(self, input_shape):
        
        model = Sequential()
        model.add(keras.layers.SimpleRNN(32, input_shape=input_shape, name='RNN'))
        model.add(Dense(2, activation='softmax', name='Output'))
        
        optimizer = keras.optimizers.Adam(lr=0.001)
        model.compile(optimizer=optimizer,
                  loss=keras.losses.binary_crossentropy,
                  metrics=["accuracy"])
        
        self.model = model
    
    def fit(self, x_tr, y_tr, x_te, y_te, epochs, batch_size):
        return self.model.fit([x_tr], y_tr, epochs=epochs, batch_size=batch_size, validation_data=([x_te], y_te))          
        
    def predict(self, x):
        return self.model.predict_classes([x])
    
    def predict_proba(self, x):
        return self.model.predict([x])
    
    
    
class LSTMModel(BaseModel):
    def __init__(self, input_shape):
        
        model = Sequential()
        model.add(keras.layers.LSTM(32, input_shape=input_shape, name='LSTM'))
        model.add(Dense(2, activation='softmax', name='Output'))
        
        optimizer = keras.optimizers.Adam(lr=0.001)
        model.compile(optimizer=optimizer,
                  loss=keras.losses.binary_crossentropy,
                  metrics=["accuracy"])
        
        self.model = model
    
    def fit(self, x_tr, y_tr, x_te, y_te, epochs, batch_size):
        return self.model.fit([x_tr], y_tr, epochs=epochs, batch_size=batch_size, validation_data=([x_te], y_te))       
        
    def predict(self, x):
        return self.model.predict_classes([x])
    
    def predict_proba(self, x):
        return self.model.predict([x])
    
    
class GRUModel(BaseModel):
    def __init__(self, input_shape):
        
        model = Sequential()
        model.add(keras.layers.GRU(32, input_shape=input_shape, name='GRU'))
        model.add(Dense(2, activation='softmax', name='Output'))
        
        optimizer = keras.optimizers.RMSprop(lr=0.001)
        model.compile(optimizer=optimizer,
                  loss=keras.losses.categorical_crossentropy,
                  metrics=["accuracy"])
        
        self.model = model
    
    def fit(self, x_tr, y_tr, x_te, y_te, epochs, batch_size):
        return self.model.fit([x_tr], y_tr, epochs=epochs, batch_size=batch_size, validation_data=([x_te], y_te))       
        
    def predict(self, x):
        return self.model.predict_classes([x])
    
    def predict_proba(self, x):
        return self.model.predict([x])