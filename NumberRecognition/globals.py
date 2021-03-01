########## Global Imports ##########
import os # paths
import numpy as np # randomness

########## Internal Imports ##########

########## Folder Names ##########
root_dir_name = "NeuralNetwork_Projects"
project_dir_name = "NumberRecognition"
resource_dir_name = "resources"
testing_data_dir_name = "TestingSet"
testing_data_images_dir_name = testing_data_dir_name + "\\Images"
training_data_dir_name = "TrainingSet"
training_data_images_dir_name = training_data_dir_name + "\\Images"
sample_submission_dir_name = "SampleSubmission"


########## File Names ##########
testing_data_filename = "TestingSet.csv"
training_data_filename = "TrainingSet.csv"
sample_submission_data_filename = "sample_submission.csv"


########## Paths ##########
nn_root_dir_path = os.path.abspath('') # Absolute root dir is "<...>\NeuralNetwork_Projects"
cur_project_dir_path = os.path.join(nn_root_dir_path, project_dir_name) # Saving the current project directory to project directory name
cur_resources_dir_path = os.path.join(cur_project_dir_path, resource_dir_name) # Saving the directory for all the resources for this project
testing_data_dir_path = os.path.join(cur_resources_dir_path, testing_data_dir_name) # Saving the directory location for the testing data
testing_data_images_dir_path = os.path.join(cur_resources_dir_path, testing_data_images_dir_name) # Saving the directory location for the testing data images
testing_data_filepath = os.path.join(testing_data_dir_path, testing_data_filename) # Saving the path to the location of the testing data set
training_data_dir_path = os.path.join(cur_resources_dir_path, training_data_dir_name) # Saving the directory location for the training data
training_data_images_dir_path = os.path.join(cur_resources_dir_path, training_data_images_dir_name) # Saving the directory location for the training data images
training_data_filepath = os.path.join(training_data_dir_path, training_data_filename) # Saving the path to the location of the training data set
sample_submission_data_dir_path = os.path.join(cur_resources_dir_path, sample_submission_dir_name) # Saving the directory location for the sample submission data
sample_submission_data_filepath = os.path.join(sample_submission_data_dir_path, sample_submission_data_filename) # Saving the path to the location of the sample submission data set


########## Neural Network Settings and Variables ##########
seed = 1337 # Create a seed to stop potential randomness in the model when training
rng = np.random.default_rng(seed=seed)

