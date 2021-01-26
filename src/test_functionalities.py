''' Here we will write down the test cases'''

import config
from sklearn.preprocessing import MinMaxScaler

class TestFunctionalities:

    def test_file_delimiter(self):
        '''
        Check if the file delimiter matches with what is expected
        :return:
        '''
        return

    def test_dataset_shape(self):
        '''
        Check if the number of instances x number of features
        matches with what is provided in the config file
        :return:
        '''
        return

    def test_file_exists(self):
        '''
        Check if the cleaned files exists at the expected location
        :return:
        '''
        return

    def test_null_check(self):
        '''
        Check if there are any null values in the clean datasets
        :return:
        '''
        return

    def test_minmax_scaler(self):
        '''
        test if the minmax scaler restricts the values between 0-1
        :return:
        '''
        return