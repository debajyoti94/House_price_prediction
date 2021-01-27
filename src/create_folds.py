''' Here we will write code for KFold cross validation.
Not using Stratified Kfold CV as this is a regression problem and
not a classification one'''

# import modules here
import config
from sklearn.model_selection import KFold

# kfold cv code here
class CreateFolds:

    def get_kfolds(self, dataset):
        '''
        Get fold value on the kfold column
        :param dataset: input dataset
        :return: dataset with kfold values
        '''

        kf = KFold(n_splits=config.NUM_FOLDS,
                   shuffle=True, random_state=42)

        y = dataset[config.OUTPUT_FEATURE].values

        for fold_value, (t_,index) in enumerate(kf.split(X=dataset,
                                                           y=y)):
            # t_ = lists of all indices of the dataset
            # y_index = indices which belong to that fold

            dataset.loc[index, config.KFOLD_COLUMN_NAME] = fold_value


        return dataset