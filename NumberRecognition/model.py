########## Global Imports ##########
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

########## Internal Imports ##########
from data import Data

class Model:
    def __init__(self, epochs:int, input_num_units:int, hidden_num_units:int, hidden_layers:int, output_num_units:int, batch_size:int, model_optimizer:str):
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
        model.add(tf.keras.layers.Input(shape=(784),)) # set the input for the model to the number of pixels in the flattened images (28x28 pixels = 784 pixels)
        model.add(tf.keras.layers.Dense(self._hidden_num_units, activation='relu')) # set the first layer to be a dense layer with an output of 50 units
        model.add(tf.keras.layers.Dense(self._output_num_units, activation="softmax")) # set the final layer to be a dense layer with an output of a (10,1) vector
        model.summary() # Output the structure of the model for debugging
        return model
        
    def train_model(self, training_x, training_y, val_x, val_y):
        self._trained_model = self._model.fit(training_x, training_y, epochs=self._epochs, batch_size=self._batch_size, validation_data=(val_x, val_y))