########## Imports ##########
import pylab
import os
import numpy as np
import pandas as pd
from matplotlib.pyplot import imread
import keras

from globals import (rng, training_data_filepath, testing_data_filepath, sample_submission_data_filepath, training_data_images_dir_path)
from errors import (failed_to_find_data_error, failed_to_find_training_data_error, failed_to_find_testing_data_error, failed_to_find_sample_submission_data_error)

# Read data from a specified file location, expecting to read from a csv
def get_data_from_csv(csv_filepath) -> pd.DataFrame:
    print("Reading data from: " + csv_filepath)

    if os.path.exists(csv_filepath):
        print("file path: " + csv_filepath + " exists!")
    else:
        print("file path: " + csv_filepath + " DOES NOT exist!")
        raise RuntimeError(failed_to_find_data_error)

    read_data = pd.read_csv(csv_filepath)
    return read_data


# Convert image to 32-bit float numpy array
def convert_image_to_numpy_array(image_name, image_location):
    image_path = os.path.join(image_location, image_name)
    img = imread(image_path)
    converted_image = img.astype('float32')
    return converted_image

# Convert images from a specified location into an array of 32-bit float numpy arrays
def convert_images_to_numpy_array(image_data, image_location):
    converted_image_array = []
    for image_name in image_data.filename:
        converted_image = convert_image_to_numpy_array(image_name, image_location)
        converted_image_array.append(converted_image)

    return converted_image_array

# Splits Data into training/validation sets based off the percentage passed in
def split_data(converted_image_array, image_data, split_percentage):
    # prevent the split_percentage variable from exceeding a value greater than 1
    if(split_percentage > 1):
        print("(func: split_data): split_percentage exceeded 100%, capping to 100%")
        split_percentage = 1

    split_train_set_x = np.stack(converted_image_array)
    split_train_set_x /= 255.0
    split_train_set_x = split_train_set_x.reshape(-1, 784).astype('float32')

    split_train_set_y = keras.utils.to_categorical(image_data.label.values)

    split_size = int(split_train_set_x.shape[0]*split_percentage)

    split_train_set_x, val_x = split_train_set_x[:split_size], split_train_set_x[split_size:]
    split_train_set_y, val_y = split_train_set_y[:split_size], split_train_set_y[split_size:]
    image_data.label.iloc[split_size:]

    return split_train_set_x, val_x, split_train_set_y, val_y

