########## Global Imports ##########
import os # paths

# Neural Network Related Imports
import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.keras import layers

########## Internal Imports ##########
from data import Data
from model import Model
from globals import ( 
    nn_root_dir_path, cur_project_dir_path, cur_resources_dir_path, 
    training_data_filepath, training_data_images_dir_path, rng,
    testing_data_images_dir_path, testing_data_filepath
)

def main():
    # debugging:
    print("nn_root_dir: " + nn_root_dir_path)
    print("cur_project_dir: " + cur_project_dir_path)
    print("cur_resources_dir: " + cur_resources_dir_path)

    training_data = Data(training_data_filepath, training_data_images_dir_path, .3)
    adam_model = Model(5, 784, 50, 1, 10, 128, 'adam')
    adam_model.train_model(training_data.get_training_x(), training_data.get_training_y(), training_data.get_training_val_x(), training_data.get_training_val_y())
    #testing_data = Data(testing_data_filepath, testing_data_images_dir_path) # Not Needed in order to build, train and validate the model
    return

if __name__ == "__main__":
    main()