''' Here we will write the code for training and testing the model '''

import config
import create_folds
import feature_engg

import argparse

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model


# training function here
def run(dataset, fold):
    '''

    :param dataset:
    :param fold:
    :return: model trained
    '''
    return model

# inference stage here
def inference_stage(dataset, model):
    '''

    :param dataset:
    :param model:
    :return:
    '''
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--clean", type=str,
                        help='Please provide argument \"--clean dataset\" '
                             'to clean raw train and test set.')

    parser.add_argument("--train", type=str,
                        help='Please provide argument \"--train kfold\" '
                             'to train the model.')

    parser.add_argument("--test", type=str,
                        help='Please provide argument \"--test inference\" '
                             'to test the model on test set.')
    args = parser.parse_args()

    # creating dump load object
    dl_obj = feature_engg.DumpLoadFile()

    if args.clean == 'dataset':

        # load the raw datasets
        raw_train = dl_obj.load_file(config.RAW_TRAIN_DATASET)
        raw_test = dl_obj.load_file(config.RAW_TEST_DATASET)

        # get clean datasets
        fr_obj = feature_engg.FeatureEngg()
        X_train, y_train = fr_obj.cleaning_data(raw_train, dataset_type='TRAIN')
        X_test, y_test = fr_obj.cleaning_data(raw_test, dataset_type='TEST')

        # print(X_train.shape,y_train.shape)
        # print(X_test.shape, y_test.shape)

        # dump these objects
        dl_obj.dump_file(config.CLEAN_TRAIN_DATASET, X_train, y_train)
        dl_obj.dump_file(config.CLEAN_TEST_DATASET, X_test, y_test)

    elif args.train == 'kfold':
        pass

    elif args.test == 'inference':
        pass