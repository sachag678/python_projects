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

# This is module contains the function signatures for data preprocessing on the abalone
# dataset. The code is heavily commented to allow you to follow along easily.

# Please report any bugs you find using @yazabi, or email us at contact@yazabi.com.

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion

filepath = 'abalone.data.txt'

columns = ['Sex','Length','Diameter','Height','Whole weight',
'Shucked weight','Viscera weight','Shell weight','Rings']

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attriubte_names):
        self.attriubte_names = attriubte_names
    def fit(self,X, y=None):
        return self
    def transform(self, X):
        return X[self.attriubte_names].values

def load_dataset(filepath):
    """Load in the data from the txt or csv file you've downloaded.

    inputs:

    :filepath: a string containing the filepath to the csv file with the
    raw data.

    returns:

    :dataset: a Pandas dataframe containing the data in the csv file located
    at filepath. It should have column names corresponding to the names of the
    variables in the abalone dataset.
    """

    abalone_data = pd.read_csv(filepath)
    abalone_data.columns = columns

    return abalone_data

def preprocess_dataset(dataset):
    """Rescales the real-valued features in dataset such that they take on
    values between 0 and 1, and converts the categorical M/F/I Sex label to
    one-hot form.

    params:

    :dataset: a Pandas DataFrame containing the raw data as loaded by the
    load_dataset() function.

    returns: the preprocessed dataset as a Pandas DataFrame
    """
    abalone_labels = dataset["Rings"]
    abalone_data = dataset.drop("Rings", axis =1)

    num_attributes =  list(abalone_data.drop("Sex", axis=1))
    cat_attributes = ["Sex"]

    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attributes)),
        ('min_max_scaler', MinMaxScaler())
        ])

    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attributes)),
        ('encoder', LabelBinarizer())
        ])

    full_pipeline = FeatureUnion(transformer_list =[
        ('cat_pipeline', cat_pipeline),
        ('num_pipeline', num_pipeline),  
        ])

    scaled_data = full_pipeline.fit_transform(abalone_data)
    
    prepared_data = pd.DataFrame(scaled_data)
    
    new_columns = ["Sex_Male", "Sex_Female", "Sex_Infant"] + columns[1:8]

    prepared_data.columns = new_columns
    prepared_data["Rings"] = abalone_labels

    # prepared_data = prepared_data.drop("Sex_Infant", axis=1)
    # prepared_data = prepared_data.drop("Sex_Male", axis=1)
    # prepared_data = prepared_data.drop("Sex_Female", axis=1)
    # prepared_data = prepared_data.drop("Diameter", axis=1)
    # prepared_data = prepared_data.drop("Whole weight", axis=1)
    # prepared_data = prepared_data.drop("Shucked weight", axis=1)
    # prepared_data = prepared_data.drop("Viscera weight", axis=1)

    return prepared_data

def split_train_and_test(dataset,test_frac=0.25):
    """Splits the preprocessed dataset into a training subset and a testing
    subset.

    params:

    :dataset: a Pandas DataFrame containing the preprocessed data generated by
    the preprocess_dataset() function.

    :test_frac: the fraction of the total dataset to be reserved for the testing
    set.

    returns: train_data and test_data, two Pandas DataFrames respectively
    containing the training and testing data (in shuffled order).
    """

    train_data, test_data = train_test_split(dataset,test_size = test_frac, random_state = 42)

    return train_data, test_data

def split_inputs_and_labels(dataset):
    """Separates dataset into its input and label components.

    params:

    :dataset: a Pandas DataFrame containing preprocessed data, which is to be
    separated into its inputs and labels.

    returns: inputs and labels, two pandas DataFrames respectively containing
    all the input columns (all columns except Rings), and the Rings column only.
    """

    labels = dataset["Rings"]
    
    inputs = dataset.drop("Rings", axis =1)

    return inputs,labels

def generate_data():
    """Loads the raw data contained in abalone.data.txt using the load_dataset()
    function, preprocesses the data using the preprocess_dataset() function,
    splits it into testing and training sets using the split_train_and_test()
    function, and finally generates separate testing/training inputs and labels
    using the split_inputs_and_labels() function.

    params: None

    returns: the testing and training inputs and labels as Pandas DataFrames.
    """

    train_data, test_data = split_train_and_test(preprocess_dataset(load_dataset(filepath)))

    train_input, train_labels = split_inputs_and_labels(train_data)
    test_input, test_labels = split_inputs_and_labels(test_data)

    return train_input, train_labels, test_input, test_labels

if __name__ == '__main__':
    train_input, train_labels, test_input, test_labels = generate_data()
