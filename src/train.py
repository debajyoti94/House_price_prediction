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
    train_set = dataset[dataset.kfold != fold]
    validation_set = dataset[dataset.kfold == fold]

    X_train = train_set.drop([config.OUTPUT_FEATURE, config.KFOLD_COLUMN_NAME],
                             axis=1,
                             inplace=False).values
    y_train = train_set[config.OUTPUT_FEATURE].values

    X_valid = validation_set.drop([config.OUTPUT_FEATURE, config.KFOLD_COLUMN_NAME],
                             axis=1,
                             inplace=False).values
    y_valid = validation_set[config.OUTPUT_FEATURE].values

    model = Sequential()
    model.add(Dense(config.NUM_FEATURES, activation='relu'))
    model.add(Dense(config.NUM_FEATURES, activation='relu'))
    model.add(Dense(config.NUM_FEATURES, activation='relu'))
    model.add(Dense(config.NUM_FEATURES, activation='relu'))
    model.add(Dense(1))

    optimizer = Adam(lr=config.LR)
    model.compile(optimizer=optimizer, loss='mse')

    model.fit(x=X_train, y=y_train, batch_size=config.BATCH_SIZE,
              epochs=config.NUM_EPOCHS, validation_data=(X_valid,y_valid)
              )

    train_loss = model.history.history['loss']
    valid_loss = model.history.history['val_loss']

    return model, train_loss, valid_loss

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
        train_dataset = fr_obj.cleaning_data(raw_train, dataset_type='TRAIN')
        test_dataset = fr_obj.cleaning_data(raw_test, dataset_type='TEST')

        # print(X_train.shape,y_train.shape)
        # print(X_test.shape, y_test.shape)

        # dump these objects
        dl_obj.dump_file(config.CLEAN_TRAIN_DATASET, train_dataset)
        dl_obj.dump_file(config.CLEAN_TEST_DATASET, test_dataset)

    elif args.train == 'kfold':

        #  load the train set
        train_set = dl_obj.load_file(config.CLEAN_TRAIN_DATASET)

        for fold_value in range(config.NUM_FOLDS):
            model, train_loss, validation_loss = run(train_set[0],
                                                     fold_value)

            dl_obj.dump_file(str(config.MODEL_NAME)+str(fold_value)+".pickle",
                             model)
            dl_obj.dump_file( "../results/train_loss_"+str(fold_value)+".pickle",
                             train_loss)
            dl_obj.dump_file("../results/validation_loss_" + str(fold_value) + ".pickle",
                             validation_loss)


    elif args.test == 'inference':

        # load model

        # load test set
        pass