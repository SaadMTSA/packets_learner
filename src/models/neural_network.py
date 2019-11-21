import numpy as np
import keras
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from .base import BaseModel
from skopt import BayesSearchCV
from keras.models import Sequential, Model, Input
from keras.layers import Dense, Activation, Input, SimpleRNN, GRU, LSTM, concatenate, Bidirectional, TimeDistributed, Embedding

class RNNModel(BaseModel):
    def __init__(self, input_shape, n_classes, rnn_seq):
        
        inputs = [Input(shape=(rnn_seq,1)) for i in range(10)]
        rnn_units = [Bidirectional(SimpleRNN(10, recurrent_dropout=0.3, return_sequences=True))(inputs[i]) for i in range(10)]
        all_units = concatenate(rnn_units)
        out = TimeDistributed(Dense(units=n_classes, activation='softmax', name='Output'))(all_units)
        model = Model(inputs, out)
        optimizer = keras.optimizers.Adam(lr=0.01)
        model.compile(optimizer=optimizer,
                  loss=keras.losses.categorical_crossentropy,
                  metrics=["accuracy"])
        self.model = model
    
    def fit(self, x_tr, y_tr, x_te, y_te, epochs, batch_size):
        return self.model.fit(x_tr, y_tr, epochs=epochs, batch_size=batch_size, validation_data=(x_te, y_te))       
        
    def predict(self, x):
        return np.argmax(self.model.predict(x), axis=-1)
    
    def predict_proba(self, x):
        return self.model.predict(x)
    
    
    
class LSTMModel(BaseModel):
    def __init__(self, input_shape, n_classes, rnn_seq):
        
#         sd_in = Input(shape=(rnn_seq, 4), dtype=tf.float32)
#         s_in = Input(shape=(rnn_seq, 19), dtype=tf.float32)
#         n_in = Input(shape=(rnn_seq, 20), dtype=tf.float32)
#         m_in = Input(shape=(rnn_seq, 1), dtype=tf.float32)
#         sd_gru = Bidirectional(LSTM(10, recurrent_dropout=0.3, return_sequences=True))(sd_in)
#         n_gru = Bidirectional(LSTM(10, recurrent_dropout=0.3, return_sequences=True))(n_in)
#         s_gru = Bidirectional(LSTM(10, recurrent_dropout=0.3, return_sequences=True))(s_in)
#         m_gru = Bidirectional(LSTM(10, recurrent_dropout=0.3, return_sequences=True))(m_in)
#         all_gru = concatenate([sd_gru, n_gru, s_gru, m_gru])
#         out = TimeDistributed(Dense(units=n_classes, activation='softmax', name='Output'))(all_gru)
#         model = Model([sd_in, s_in, n_in, m_in], out)
#         optimizer = keras.optimizers.Adam(lr=0.001)
#         model.compile(optimizer=optimizer,
#                   loss=keras.losses.categorical_crossentropy,
#                   metrics=["accuracy"])
#         self.model = model
        inputs = [Input(shape=(rnn_seq,1)) for i in range(10)]
        rnn_units = [Bidirectional(LSTM(10, recurrent_dropout=0.3, return_sequences=True))(inputs[i]) for i in range(10)]
        all_units = concatenate(rnn_units)
        out = TimeDistributed(Dense(units=n_classes, activation='softmax', name='Output'))(all_units)
        model = Model(inputs, out)
        optimizer = keras.optimizers.Adam(lr=0.001)
        model.compile(optimizer=optimizer,
                  loss=keras.losses.categorical_crossentropy,
                  metrics=["accuracy"])
        self.model = model
    
    def fit(self, x_tr, y_tr, x_te, y_te, epochs, batch_size):
        return self.model.fit(x_tr, y_tr, epochs=epochs, batch_size=batch_size, validation_data=(x_te, y_te))       
        
    def predict(self, x):
        return np.argmax(self.model.predict(x), axis=-1)
    
    def predict_proba(self, x):
        return self.model.predict(x)
    
    
class GRUModel(BaseModel):
    def __init__(self, input_shape, n_classes, rnn_seq):
        
#         model = Sequential()
        
#         model.add(Bidirectional(GRU(80, input_shape=input_shape, recurrent_dropout=0.25, return_sequences=True)))
#         model.add(TimeDistributed(Dense(units=n_classes, activation='softmax', name='Output')))
        
#         sd_in = Input(shape=(rnn_seq, 4), dtype=tf.float32)
#         s_in = Input(shape=(rnn_seq, 19), dtype=tf.float32)
#         n_in = Input(shape=(rnn_seq, 20), dtype=tf.float32)
#         m_in = Input(shape=(rnn_seq, 1), dtype=tf.float32)
#         sd_gru = Bidirectional(GRU(10, recurrent_dropout=0.3, return_sequences=True))(sd_in)
#         n_gru = Bidirectional(GRU(10, recurrent_dropout=0.3, return_sequences=True))(n_in)
#         s_gru = Bidirectional(GRU(10, recurrent_dropout=0.3, return_sequences=True))(s_in)
#         m_gru = Bidirectional(GRU(10, recurrent_dropout=0.3, return_sequences=True))(m_in)
#         all_gru = concatenate([sd_gru, n_gru, s_gru, m_gru])
#         out = TimeDistributed(Dense(units=n_classes, activation='softmax', name='Output'))(all_gru)
#         model = Model([sd_in, s_in, n_in, m_in], out)
#         optimizer = keras.optimizers.Adam(lr=0.001)
#         model.compile(optimizer=optimizer,
#                   loss=keras.losses.categorical_crossentropy,
#                   metrics=["accuracy"])
#         self.model = model
        inputs = [Input(shape=(rnn_seq,1)) for i in range(10)]
        rnn_units = [Bidirectional(GRU(10, recurrent_dropout=0.3, return_sequences=True))(inputs[i]) for i in range(10)]
        all_units = concatenate(rnn_units)
        out = TimeDistributed(Dense(units=n_classes, activation='softmax', name='Output'))(all_units)
        model = Model(inputs, out)
        optimizer = keras.optimizers.Adam(lr=0.01)
        model.compile(optimizer=optimizer,
                  loss=keras.losses.categorical_crossentropy,
                  metrics=["accuracy"])
        self.model = model
        
    def fit(self, x_tr, y_tr, x_te, y_te, epochs, batch_size):
        self.X = x_tr
        return self.model.fit(x_tr, y_tr, epochs=epochs, batch_size=batch_size, validation_data=(x_te, y_te))       
        
    def predict(self, x):
        return np.argmax(self.model.predict(x), axis=-1)
    
    def predict_proba(self, x):
        return self.model.predict(x)

class CustomNNModel(BaseModel):
    def __init__(self, input_shape, n_classes, rnn_seq):
        
        input_ = Input(input_shape)
        gru = GRU(32, name='GRU')(input_)
        lstm = LSTM(32, name='LSTM')(input_)
        rnn = SimpleRNN(32, name='RNN')(input_)
        agg = concatenate([gru, lstm, rnn])
        output = Dense(n_classes, activation='softmax', name='Output')(agg)
        
        model = Model([input_], output)
        
        optimizer = keras.optimizers.RMSprop(lr=0.01)
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