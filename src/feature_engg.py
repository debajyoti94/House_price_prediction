''' Here we will write code for feature engineering the raw dataset'''

#  import modules here
# import sys
# sys.path.append('/home/deb/untitled/DSDJ/DSDJ workshops/Linear_Regression/My_Code/House_price_prediction/src')

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import abc
import pickle

import config
import create_folds

# create abc class here
class MustHaveForFeatureEngg:

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def drop_features(self):
        '''
        Pass a list of features that you want to drop
        :return: dataframe with the dropped features
        '''
        return

    @abc.abstractmethod
    def cleaning_data(self):
        '''
        Clean the given input data
        :return: cleaned data
        '''
        return


# create feature engg class here
class FeatureEngg(MustHaveForFeatureEngg):

    def cleaning_data(self, data, dataset_type):
        '''
        Pre-processing the raw data and providing a clean version.
        :param data: input dataset
        :param dataset_type: TRAIN or TEST flag,
        this will be used for minmax scaling
        :return: cleaned dataset
        '''
        print(data.shape)

        # extracting year and month from date feature
        data['date'] = pd.to_datetime(data['date'])
        data['year'] = data['date'].apply(lambda date: date.year)
        data['month'] = data['date'].apply(lambda date: date.month)

        # dropping the features we do not need
        data = self.drop_features(data, config.FEATURES_TO_DROP)
        print(data.shape)
        # X_features = data.drop(config.OUTPUT_FEATURE, axis=1, inplace=False)
        # y_target = data[config.OUTPUT_FEATURE]

        scaler = MinMaxScaler()

        if dataset_type == 'TRAIN':
            # scaling data here
            scaler.fit_transform(data.drop(config.OUTPUT_FEATURE, axis=1, inplace=False))
            print(data.max())
            # applying kfold cv labels here
            data[config.KFOLD_COLUMN_NAME] = -1
            kfold_obj = create_folds.CreateFolds()
            data = kfold_obj.get_kfolds(data)

            # pickle the scaler
            dl_obj = DumpLoadFile()
            dl_obj.dump_file('../models/scaler.pickle', scaler)

            print(data.columns)
        elif dataset_type == 'TEST':
            dl_obj = DumpLoadFile()
            data_scaler = dl_obj.load_file('../models/scaler.pickle')
            data_scaler[0].transform(data.drop(config.OUTPUT_FEATURE, axis=1, inplace=False))

        return data


    def drop_features(self, data_df, features_to_drop):
        '''
        Function to drop the features that are passed
         in the parameter
        :param data_df: The input dataframe
        :param features_to_drop: list of features that you want to drop
        :return: dataframe with dropped features
        '''
        data_df = data_df.drop(features_to_drop,
                            axis=1, inplace=False)

        return data_df

# class for dumping and loading pickled files
class DumpLoadFile:

    def load_file(self, filename):
        """
        For loading the pickled files
        :param filename: the file that you want to load
        :return: the loaded file
        """
        with open(filename, "rb") as pickle_handle:
            return pickle.load(pickle_handle)

    def dump_file(self, filename, *file):
        """
        Pickle a file
        :param *file: files that you want to pickle
        :param filename: filename for the pickled file
        :return: nothing
        """
        with open(filename, "wb") as pickle_handle:
            pickle.dump(file, pickle_handle)




