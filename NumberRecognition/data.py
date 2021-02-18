########## Imports ##########
from preprocessing import get_data_from_csv, convert_images_to_numpy_array, split_data
from errors import failed_to_load_data_for_class
import pandas as pd


class Data:
    def __init__(self, csv_data_loc, images_dir, split_data_percentage):
        try:
            self._raw_data = get_data_from_csv(csv_data_loc)
            self._converted_data = convert_images_to_numpy_array(self._raw_data, images_dir) # TODO: Remove .head() after testing and verifying if the functionality is correct
            self._training_set_x, self._validation_set_x, self._training_set_y, self._validation_set_y = split_data(self._converted_data, self._raw_data, split_data_percentage) # TODO: Remove .head() after testing and verifying if the functionality is correct
        except RuntimeError:
            print(failed_to_load_data_for_class)
            raise

    def get_training_set_x(self):
        return self._training_set_x

    def get_training_set_y(self):
        return self._training_set_y

    def get_validation_set_x(self):
        return self._validation_set_x

    def get_validation_set_y(self):
        return self._validation_set_y

    def get_raw_data(self):
        return self._raw_data
