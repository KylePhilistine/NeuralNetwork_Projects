from data import Data

from keras.models import Sequential
from keras.layers import Dense

import tensorflow as tf

class Model:
    def __init__(self, epochs:int, input_num_units:int, hidden_num_units:int, hidden_layers:int, output_num_units:int,batch_size:int, model_optimizer:str):
        self._epochs = epochs
        self._input_num_units = input_num_units
        self._hidden_num_units = hidden_num_units
        self._hidden_layers = hidden_layers
        self._output_num_units = output_num_units
        self._batch_size = batch_size

        self._model = self._create_model()
        self._model.compile(loss='categorical_crossentropy', optimizer=model_optimizer, metrics=['accuracy'])

    def _create_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.Input(shape=(self._input_num_units,)))
        #model.add(tf.keras.layers.Dense(self._output_num_units, activation='relu'))
        return model
        
    def train_model(self, training_set_x, validation_set_x, training_set_y, validation_set_y):
        print("training_set_x:" + str(training_set_x.size))
        print("training_set_y:" + str(training_set_y.size))
        self._trained_model = self._model.fit(training_set_x, training_set_y, epochs=self._epochs, batch_size=self._batch_size, validation_split=.7)