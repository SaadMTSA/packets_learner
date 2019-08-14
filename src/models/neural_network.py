import numpy as np
import keras

from sklearn.linear_model import LogisticRegression
from .base import BaseModel
from skopt import BayesSearchCV
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, SimpleRNN, GRU, LSTM, concatenate

class RNNModel(BaseModel):
    def __init__(self, input_shape):
        
        model = Sequential()
        model.add(SimpleRNN(32, input_shape=input_shape, name='RNN'))
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
        model.add(LSTM(32, input_shape=input_shape, name='LSTM'))
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
        model.add(GRU(32, input_shape=input_shape, name='GRU'))
        model.add(Dense(2, activation='softmax', name='Output'))
        
        optimizer = keras.optimizers.RMSprop(lr=0.001)
        model.compile(optimizer=optimizer,
                  loss=keras.losses.categorical_crossentropy,
                  metrics=["accuracy"])
        
        self.model = model
    
    def fit(self, x_tr, y_tr, x_te, y_te, epochs, batch_size):
        self.X = x_tr
        return self.model.fit([x_tr], y_tr, epochs=epochs, batch_size=batch_size, validation_data=([x_te], y_te))       
        
    def predict(self, x):
        return self.model.predict_classes([x])
    
    def predict_proba(self, x):
        return self.model.predict([x])

class CustomNNModel(BaseModel):
    def __init__(self, input_shape):
        
        input_ = Input(input_shape)
        gru = GRU(32, name='GRU')(input_)
        lstm = GRU(32, name='LSTM')(input_)
        rnn = GRU(32, name='RNN')(input_)
        agg = concatenate([gru, lstm, rnn])
        output = Dense(2, activation='softmax', name='Output')(agg)
        
        model = Model([input_], output)
        
        optimizer = keras.optimizers.RMSprop(lr=0.001)
        model.compile(optimizer=optimizer,
                  loss=keras.losses.categorical_crossentropy,
                  metrics=["accuracy"])
        
        self.model = model
        
    def fit(self, x_tr, y_tr, x_te, y_te, epochs, batch_size):
        return self.model.fit([x_tr], y_tr, epochs=epochs, batch_size=batch_size, validation_data=([x_te], y_te))       
        
    def predict(self, x):
        if hasattr(self.model, 'predict_classes'):
            return self.model.predict_classes([x])
        else:
            return np.argmax(self.predict_proba(x), axis=-1)
    
    def predict_proba(self, x):
        return self.model.predict([x])