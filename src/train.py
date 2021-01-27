''' Here we will write the code for training and testing the model '''

import config
import create_folds
import feature_engg

import argparse

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
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
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1))

    optimizer = Adam(lr=config.LR)
    model.compile(optimizer=optimizer, loss='mse')

    # defining early stopping in case model overfits
    # mode= min which means you want to minimize the val loss
    # patience=25 wait for 25 epochs after val loss starts increasing
    early_stop = EarlyStopping(monitor='val_loss', mode='min',
                               patience=25)

    model.fit(x=X_train, y=y_train, batch_size=config.BATCH_SIZE,
              epochs=config.NUM_EPOCHS, validation_data=(X_valid,y_valid),
              callbacks=[early_stop]
              )

    losses = model.history.history

    return model, losses

# inference stage here
def inference_stage(dataset, model):
    '''
    Run model on test set and get model performance
    :param dataset: test set
    :param model: model that is trained
    :return:
    '''
    X_test = dataset.drop(config.OUTPUT_FEATURE, axis=1, inplace=False).values
    y_test = dataset[config.OUTPUT_FEATURE].values

    print(model.evaluate(X_test, y_test))


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
            model, losses = run(train_set[0],fold_value)

            dl_obj.dump_file(str(config.MODEL_NAME)+str(fold_value)+".pickle",
                             model)
            dl_obj.dump_file( "../results/loss_"+str(fold_value)+".pickle",
                             losses)



    elif args.test == 'inference':

        # load model
        model = dl_obj.load_file(config.BEST_MODEL)

        # load test set
        test_set = dl_obj.load_file(config.CLEAN_TEST_DATASET)

        # calling the inference stage function
        inference_stage(dataset=test_set[0], model=model[0])
