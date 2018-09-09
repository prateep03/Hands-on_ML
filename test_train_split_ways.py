from six.moves import urllib

import numpy as np
import os
import tarfile
import pandas as pd
import hashlib
import pprint

DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
HOUSING_PATH = 'datasets/housing'
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

def fetch_housing_data(housing_url = HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path = HOUSING_PATH):
    # fetch_housing_data(housing_path=housing_path)
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

'''
split train and test set by random permutation of indices
'''
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

'''
split train test set by ids, to make the test set consistent
'''
def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_colum, hash = hashlib.md5):
    ids = data[id_colum]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

'''
Label Encoders
 - cateogorical to integer
 - integer to one-hot 
'''
def label_encoders(housing):
    '''
        LabelEncododer : categorical to numerical
        '''
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    housing_cat = housing["ocean_proximity"]
    housing_cat_enc = encoder.fit_transform(housing_cat)
    print(encoder.classes_)

    '''
    OneHotEncoder : numerical to one-hot encoding (sparse matrix)
    '''

    from sklearn.preprocessing import OneHotEncoder

    encoder = OneHotEncoder()
    housing_cat_1hot = encoder.fit_transform(housing_cat_enc.reshape(-1, 1))
    print("Type of housing_cat_1hot : ", type(housing_cat_1hot))
    print(housing_cat_1hot.toarray())

    from sklearn.preprocessing import LabelBinarizer

    encoder = LabelBinarizer(sparse_output=False)
    housing_cat_bin = encoder.fit_transform(housing_cat)
    print(type(housing_cat_bin))  # numpy.ndarray -> dense array if sparse_output set to False

'''
Custom Transformers
  - transforming data
  - TransformerMixin base class gives fit(), transform() and fit_transform() methods
  - BaseEstimator base class gives get_params() and set_params() methods
'''

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix , bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kwargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self # nothing to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

'''
Feature Scaling
  i) min-max scaling (MinMaxScaler)
    - subtract min value and divide by max minus min. 
    - feature_range parametere that lets you change the range if you dont want [0,1].
  ii) standardization (StandardScaler)
    - subtract mean value (so standardized data always has zero mean)
    - divide by the variance so resulting distribution has unit variance 
'''

def feature_scalers(housing):

    from sklearn.preprocessing import Imputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    num_pipeline = Pipeline([
        ('imputer', Imputer(strategy='median')),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

    housing_num = housing.drop("ocean_proximity", axis=1)
    housing_num_tr = num_pipeline.fit_transform(housing_num)
    print(housing_num_tr)

'''
class to feed a Pandas DataFrame directly into Pipeline
'''

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

class CustomLabelBinarizer(TransformerMixin):
    def __init__(self, sparse_output = False):
        self.sparse_output = sparse_output
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        enc = LabelBinarizer(sparse_output=self.sparse_output)
        return enc.fit_transform(X)

def feature_scalers_on_DataFrame(housing):

    from sklearn.preprocessing import Imputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, LabelBinarizer

    housing_num = housing.drop("ocean_proximity", axis=1)

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(attribute_names=num_attribs)),
        ('imputer', Imputer(strategy='median')),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(attribute_names=cat_attribs)),
        ('label_binarizer', CustomLabelBinarizer()), # pipeline assumes LabelBinarizer(scikit-learn 0.19.0) has 3 positional arguments, but it takes 2 only
    ])

    from sklearn.pipeline import FeatureUnion

    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

    housing_prepared = full_pipeline.fit_transform(housing)
    pprint.pprint("housing_prepared shape = %s" % repr(housing_prepared.shape))

    return housing_prepared, full_pipeline

if __name__ == "__main__":
    housing = load_housing_data()
    # housing_with_id = housing.reset_index() # adds an `index` column
    # # train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
    # # print(len(train_set), "train + ", len(test_set), "test")

    # feature_scalers(housing)

    feature_scalers_on_DataFrame(housing)