# Copyright (c) 2017 Yazabi Predictive Inc.

#################################### MIT License ####################################
#                                                                                   #
# Permission is hereby granted, free of charge, to any person obtaining a copy      #
# of this software and associated documentation files (the "Software"), to deal     #
# in the Software without restriction, including without limitation the rights      #
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell         #
# copies of the Software, and to permit persons to whom the Software is             #
# furnished to do so, subject to the following conditions:                          #
#                                                                                   #
# The above copyright notice and this permission notice shall be included in all    #
# copies or substantial portions of the Software.                                   #
#                                                                                   #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR        #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,          #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE       #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER            #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,     #
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE     #
# SOFTWARE.                                                                         #
#                                                                                   #
#####################################################################################

# This module contains a class template for model training and prediction on the Communities and
# Crime dataset using XGBoost. The code is heavily commented to allow you to follow along easily.

import data_preprocessing as dp
import pandas as pd
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

cv_params = {'max_depth': [3,5,7], 'min_child_weight': [3,5,7], 'learning_rate': [0.1,0.01],
            'subsample': [0.5,0.7,0.9], 'n_estimators': [100],
            'colsample_bytree': [0.6], 'objective': ['reg:logistic'], 'gamma': [0.5],
            'reg_lambda': [0.5], 'base_score': [0.5], 'booster': ['gbtree']}


def find_best_parameters(train_inputs, train_labels, cv_params):
    """Runs a grid search over parameter space by using GridSearchCV on the
    xgb.XGBRegressor() model, and identifies the best parameters for the model
    among the options in cv_params.

    params:

    :train_inputs: a Pandas DataFrame providing the inputs to the learning
    algorithm.
    :train_labels: a Pandas DataFrame providing the labels to the learning
    algorithm.
    :cv_params: a dict object containing the parameter values to be tested.

    returns: optimized_params, a dictionary object containing the optimal
    parameter values identified by the grid search.
    """

    pass

def find_optimal_num_trees(optimized_params,dmat,num_boost_round=3000,
                            early_stopping_rounds=1000):
    """Determines the best number of trees to use in order to minimize
    cross-validation error. Hint: you'll want to use the xgb.cv() function
    for this.

    params:

    :optimized_params: a dictionary object providing the parameter values that
    have been found to be optimal based on a previously executed grid search.
    :dmat: an xgboost DMatrix object in the format
    xgb.DMatrix(train_inputs, train_labels['ViolentCrimesPerPop']).
    :num_boost_round: an int providing the number of boosting iterations used
    by the algorithm.
    :early_stopping_rounds: an int providing the number of consecutive boosting
    rounds over which the validation loss must not decrease before the algorithm
    stops.

    returns: optimal_num_trees, an int indicating the optimal number of decision
    trees to be added together.
    """

    pass

def build_model(train_inputs, train_labels, cv_params=cv_params, drop_threshold=0.1):
    """Builds and trains the XGBoost model using the grid of parameters provided
    as input.

    params:

    :train_inputs: a Pandas DataFrame containing the training input data.
    :train_labels: a Pandas DataFrame containing the training labels.
    :cv_params: a dictionary object providing the grid of parameter values
    to be explored in the search for the optimal set of XGBoost parameters.
    :drop_threshold: a float indicating the minimum correlation between an
    input feature and the label being investigated (generally,
    'ViolentCrimesPerPop'), below which we will discard the feature.

    returns: final_gb, the XGBoost model trained using the optimized parameters
    from cv_params.
    """

    pass

def assess_model(model, test_inputs, test_labels):
    """Calculates the mean absolute error that results from the predictions
    of the model, taken as input, based on the testing data provided.
    Suggestion: first convert the test_inputs into an xgb.DMatrix(), and then
    use scikit-learn's mean_absolute_error function, which has been imported
    for you at the top of the script.

    params:

    :model: an XGBoost model that has been trained on an optimized set of
    parameters.
    :test_inputs: a Pandas DataFrame providing the testing input data.
    :test_labels a Pandas DataFrame providing the testing labels.

    returns: mae, the mean absolute error obtained by applying the model
    to the testing data.
    """

    pass

def determine_importances(model):
    """Determines the importances of each feature in the model using
    the XGBoost get_fscore() function.

    params:

    :model: the trained XGBoost model.

    returns: importances, a dictionary object containing the importances
    assigned to each feature.
    """

    pass

if __name__ == "__main__":
    #In this block, write the script that you want to have run automatically
    #as soon as this module is called. You'll want to do the following:
    # 1. load the preprocessed training and testing data using dp.generate_data()
    # 2. plot the correlations matrix using dp.plot_correlations()
    # 3. build your optimized model using build_model()
    # 4. calculate and print the mean absolute error that your model obtains
    # on the testing data using assess_model()
    # 5. find the importances of each feature using importance_frame(), and
    # plot them.

    pass
