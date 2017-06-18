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

# This is module contains the function signatures for predicton functions that will work
# on the abalone dataset. The code is heavily commented to allow you to follow along
# easily.

# Please report any bugs you find using @yazabi, or email us at contact@yazabi.com.

import pandas as pd
import data_exploration as de
import data_preprocessing as dp
import pickle
from sklearn.externals import joblib

def save_model(model):
    """Saves the model provided as input in the current folder, with the
    name 'optimized_model'.

    params:

    :model: the model to be saved. This should be the optimized model that you
    obtain by testing out each algorithm type, parameter set and model_mode
    that you think is likely to work, and will be used for "in-production"
    predictions.

    returns: nothing.

    Hint: to save your model you can use the pickle package, which has already
    been imported for you. See http://scikit-learn.org/stable/modules/model_persistence.html.
    """
    joblib.dump(model,'optimized_model.pkl')
    

def load_model():
    """Loads the saved model for later use in "production mode".

    params: None

    returns: the optimized model, previously saved by the save_model() function.
    """

    return joblib.load('optimized_model.pkl')

def build_and_save_model(train_inputs, train_labels, model_type,model_mode,params):
    """Once you've played around with various model architectures and
    parameter values in data_exploration.py, you'll choose the best performing
    model_type, model_mode and params values, and use them in production. This
    function takes these inputs and saves a fully trained model, ready to
    make predictions.

    params:

    :model_type: a string indicating the model architecture you've found most
    effective. Can be any one of 'naive_bayes', 'knn', 'svm' or 'decision_tree'.

    :model_mode: a string indicating whether you found the problem was best
    treated as a classification problem or a regression problem (respectively
    denoted by the two allowable values 'classification' and 'regression').

    :params: a dict object containing the parameters you found most optimal
    for the model_type and model_mode provided.

    :train_inputs: a Pandas dataframe obtained by passing appropriately
    preprocessed training data to the split_inputs_and_labels() function in the
    data_preprocessing.py module. Columns represent the features taken as
    input by our learning models.

    :train_labels: a Pandas dataframe likewise obtained from
    split_inputs_and_labels(), corresponding to the training set's Rings values.

    returns: nothing
    """
    save_model(de.build_model(train_inputs, train_labels,params,model_mode, model_type))

def predict(inputs):
    """Predicts the Rings values for the inputs provided, based on a saved
    and pretrained model.

    params:

    :inputs: a Pandas dataframe containing the inputs whose Rings values are to
    be predicted.

    returns: the predicted Rings values.
    """

    model = load_model()
    return model.predict(inputs)

if __name__ == '__main__':
    """Loads training inputs and labels using dp.generate_data(),
    then trains and saves a model using build_and_save_model(). Finally,
    Re-loads the model using load_model, and runs evaluate_model()
    to display the corresponding confusion matrix.
    """
    model_params = {}   
    model_mode = "classification"
    model_type = "svm"

    train_inputs, train_labels, test_inputs, test_labels = dp.generate_data()
    build_and_save_model(train_inputs, train_labels, model_type,model_mode,params)
    model = load_model()
    evaluate_model(model, test_inputs, test_labels, model_mode)
