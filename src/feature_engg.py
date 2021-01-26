''' Here we will write code for feature engineering the raw dataset'''

#  import modules here
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import abc
import pickle

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

        :param data:
        :param dataset_type:
        :return:
        '''
        return


    def drop_features(self, data_df, features_to_drop):
        '''
        Function to drop the features that are passed
         in the parameter
        :param data_df: The input dataframe
        :param features_to_drop: list of features that you want to drop
        :return: dataframe with dropped features
        '''

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


# create pickle dump load class here

# writing some snippets here
df['date'] = pd.to_datetime(df['date'])

df['year'] = df['date'].apply(lambda date: date.year)
df['month'] = df['date'].apply(lambda date: date.month)

