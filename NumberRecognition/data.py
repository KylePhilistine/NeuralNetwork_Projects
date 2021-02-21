########## Imports ##########
from preprocessing import get_data_from_csv, convert_images_to_numpy_array, split_data
from errors import failed_to_load_data_for_class
import pandas as pd


class Data:
    def __init__(self, csv_data_loc, images_dir, test__split_percentage):
        try:
            self._raw_data = get_data_from_csv(csv_data_loc)
            self._converted_data = convert_images_to_numpy_array(self._raw_data, images_dir)
            self._training_x, self._training_y, self._training_val_x, self._training_val_y = split_data(self._converted_data, self._raw_data, test__split_percentage)
        except RuntimeError:
            print(failed_to_load_data_for_class)
            raise

    def get_training_x(self):
        return self._training_x

    def get_training_y(self):
        return self._training_y

    def get_training_val_x(self):
        return self._training_val_x

    def get_training_val_y(self):
        return self._training_val_y    

    def get_raw_data(self):
        return self._raw_data
