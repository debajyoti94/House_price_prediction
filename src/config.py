''' Here we will define the variables which will be used
throughout this project'''


# for the dataset
ORIGINAL_RAW_DATASET = '../input/kc_house_data.csv'

RAW_TRAIN_DATASET = '../input/raw_train_set.pickle'
RAW_TEST_DATASET = '../input/raw_test_set.pickle'

CLEAN_TRAIN_DATASET = '../input/clean_train_set.pickle'
CLEAN_TEST_DATASET = '../input/clean_test_set.pickle'

FILE_DELIMITER = ','
ENCODING_TYPE = 'UTF-8'
RAW_DATASET_SHAPE = (21267, 19)

#  for the model
NUM_FOLDS = 5
MODEL_NAME = '../models/KC_LR_MODEL_'
MODEL_LOSS = 'mse'
MODEL_OPTIMIZER = 'adam'
NUM_EPOCHS = 200