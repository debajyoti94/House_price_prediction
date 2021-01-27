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
OUTPUT_FEATURE = 'price'
FEATURES_TO_DROP = ['id', 'zipcode', 'date']

#  for the model
NUM_FEATURES = 19
BATCH_SIZE = 64
NUM_FOLDS = 5
KFOLD_COLUMN_NAME = 'kfold'
MODEL_NAME = '../models/KC_LR_MODEL_'
BEST_MODEL = '../models/KC_LR_MODEL_0.pickle'
MODEL_LOSS = 'mse'
MODEL_OPTIMIZER = 'adam'
NUM_EPOCHS = 400
LR = 0.9