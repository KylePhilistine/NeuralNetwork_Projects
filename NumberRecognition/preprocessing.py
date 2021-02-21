########## Imports ##########
import pylab
import os
import numpy as np
import pandas as pd
from matplotlib.pyplot import imread
import imageio
import keras
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

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
    print("Image Data Read in Successfully!")
    return read_data


# Convert image to 32-bit float numpy array
def convert_image_to_numpy_array(image_name, image_location):
    image_path = os.path.join(image_location, image_name)
    img = imageio.imread(image_path, as_gray=True)
    converted_image = img.astype('float32')
    return converted_image

# Convert images from a specified location into an array of 32-bit float numpy arrays
def convert_images_to_numpy_array(image_data, image_location):
    converted_raw_image_array = []
    for image_name in image_data.filename:
        converted_raw_image_data = convert_image_to_numpy_array(image_name, image_location)
        converted_raw_image_data = np.asarray(converted_raw_image_data)
        converted_raw_image_array.append(converted_raw_image_data)

    print("Coverted Images to Raw Data!")
    return converted_raw_image_array

# Splits Data into training/validation sets based off the percentage passed in
def split_data(converted_image_array, image_data, test__split_percentage):
    X = np.stack(converted_image_array)
    X /= 255.0 # normalize values to be between 0-1 (gray scale is between 0-255)
    X = X.reshape(-1, 784).astype('float32') # flatten numpy array from 1x28x28 to 1x784, where 784 is the total number of pixels for the gray scale images

    Y = image_data.iloc[:,len(image_data.columns)-1:] # use the last column as results for testing
    Y = keras.utils.to_categorical(Y) # change the Y data from strings to values

    min_max_scaler = preprocessing.MinMaxScaler()
    X_scale = min_max_scaler.fit_transform(X)

    X_train, X_val, Y_train, Y_val = train_test_split(X_scale, Y, test_size=test__split_percentage)
    return X_train, Y_train, X_val, Y_val

