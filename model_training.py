import os
import pandas as pd
import numpy as np

from test_train_split_ways import feature_scalers_on_DataFrame

HOUSING_PATH = 'datasets/housing'

def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

housing = strat_train_set.drop("median_house_value", axis = 1)

housing_prepared, full_pipeline = feature_scalers_on_DataFrame(housing)
housing_labels = strat_train_set["median_house_value"].copy()

"""
Linear Regression
"""

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = housing_prepared[:5]

print("Linear regression Predictions:", lin_reg.predict(some_data_prepared))
print("Linear regression Labels:", list(some_labels))

from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print("Linear regression rmse: %f" % lin_rmse)

'''
Linear regression Predictions: [203682.37379543 326371.39370781 204218.64588245  58685.4770482
 194213.06443039]
Linear regression Labels: [286600.0, 340600.0, 196900.0, 46300.0, 254500.0]
Linear regression rmse: 68376.642955 # --> this indicates model underfits data as all of data was used for training
'''

"""
Decision Tree Regressor
"""

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, y=housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_predictions, housing_labels)
tree_rmse = np.sqrt(tree_mse)
print("Decision Tree Regressor rmse: %f\n" % tree_rmse)

'''
Decision Tree Regressor rmse: 0.000000 # --> not good !! as model might have overfit data
'''

"""
Cross-validation
"""
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, y=housing_labels, scoring='neg_mean_squared_error', cv=10)
tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

print("Decision Tree Regression crossval scores\n")
display_scores(tree_rmse_scores)

'''
Scores: [69205.47768029 68909.48598007 71886.89243426 69110.78345046
 70224.50287071 76460.75445255 70903.58023691 70852.53758115
 76017.88323928 69196.56504549]
Mean: 71276.84629711747
Standard deviation: 2646.9538081481173
'''

lin_scores = cross_val_score(lin_reg, X=housing_prepared, y=housing_labels, scoring="neg_mean_squared_error",
                             cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
print("Linear Regression crossval scores\n")
display_scores(lin_rmse_scores)

'''
Scores: [66877.52325028 66608.120256   70575.91118868 74179.94799352
 67683.32205678 71103.16843468 64782.65896552 67711.29940352
 71080.40484136 67687.6384546 ]
Mean: 68828.99948449328
Standard deviation: 2662.761570610345
'''

"""
Random Forest Regressor
"""
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

forest_scores = cross_val_score(forest_reg, housing_prepared, y=housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
fores_rmse_scores = np.sqrt(-forest_scores)
print("Random Forest Regressor crossval scores\n")
display_scores(fores_rmse_scores)

'''
Scores: [53031.44980084 50579.09342641 52500.35525459 55184.44646788
 52440.42045415 56056.58324018 51728.37488898 49913.53378208
 56082.14671572 53020.79448027]
Mean: 53053.71985111028
Standard deviation: 2028.6341721427573
'''

"""
from sklearn.externals import joblib

joblib.dump(my_model, "my_model.pkl")
my_model_loaded = joblib.load("my_model.pkl")
"""

