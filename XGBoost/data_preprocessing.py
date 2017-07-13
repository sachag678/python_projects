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

# This module contains a class template for data preprocessing on the Communities and Crime dataset.
# The code is heavily commented to allow you to follow along easily.

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

def load_json(filepath):
    """Loads the json file stored in filepath.

    params:

    :filepath: a string providing the filepath to the json file to be loaded.

    returns: loaded_file, a list of strings loaded from the file stored at
    filepath.
    """

    return json.load(filepath)


def load_data(filepath,all_headers):
    """Loads in data from the CSV file contained in the location given by
    filepath.

    params:

    :filepath: a string containing the path to the data-containing CSV.

    returns: dataset, a Pandas DataFrame that contains the crime dataset.
    Note: this DataFrame should not contain the columns 'communityname',
    'state', 'countyCode', 'communityCode' or 'fold', since these columns are
    neither useful as inputs or labels. Hint: pd.DataFrame.drop() is a function
    you'll want to use for this.
    """

    dataset = pd.read_csv(filepath, header=None)
    dataset.columns = all_headers
    dataset = dataset.drop(['countyCode','State','fold','communityCode','communityname'], axis=1)
    return dataset

def drop_nan_cols(dataset,usable_columns,drop_threshold=230):
    """Replaces question marks '?' from the Pandas DataFrame taken as input with
    np.NaN values, and then drops any columns that contain fewer than
    drop_threshold features (you can do that using pd.Series.isnull().sum()).
    Note that the drop_threshold value will have to be greater than 221, since
    that's how many '?'s there are in the 'ViolentCrimesPerPop' column, which
    we want to analyze.

    params:

    :dataset: a Pandas DataFrame as loaded by the load_data() function.
    :usabel_columns: a list of strings, each of which provides the name of a
    column that can be used as an input or label. usable_columns will therefore
    contain all the entries in all_headers, minus 'communityname', 'state',
    'countyCode', 'communityCode' or 'fold' (which happen to be the first 5
    entries).
    :drop_threshold: the number of NaN entries above which we remove a column
    from the dataset (must be greater than 221 in order to avoid throwing
    away the 'ViolentCrimesPerPop' column, which we want to analyze)

    returns: reduced_dataset (the dataset taken as input, with any columns
    containing more than drop_threshold NaN entries dropped), and
    features_to_drop (a list of strings containing the names of any dropped
    features).
    """

   
    reduced_dataset = dataset.replace('?',np.NaN)

    features_to_drop = []
    for name in usable_columns:
        if reduced_dataset[name].isnull().sum() > drop_threshold:
            features_to_drop.append(name)
    
    reduced_dataset = reduced_dataset.drop(features_to_drop,axis=1)

    return reduced_dataset


def drop_nan_rows(dataset):
    """Removes any rows in the dataset taken as input, that contain NaN
    values.

    params:

    :dataset: a Pandas DataFrame containing the reduced_dataset returned by
    drop_nan_cols().

    return: the dataset, where any of its NaN-containing rows have been
    dropped.
    """

    dataset = dataset.dropna()
    return dataset
    

def split_train_and_test(dataset,test_frac=0.25):
    """Splits the input dataset into randomly-shuffled training and testing
    sets.

    params:

    :dataset: a Pandas DataFrame containing the total dataset, which is to be
    split into training and testing components.
    :test_frac: the fraction of samples to be reserved for testing.

    returns: train_data and test_data, the split training and testing sets as
    Pandas DataFrames.
    """

    train, test = train_test_split(dataset,test_size=test_frac,random_state=42)
    return train,test

def find_retained_labels(label_headers,features_to_drop):
    """Identifies the names of the columns that will be used as labels, and that
    haven't been dropped from the dataset yet due to having too many NaN entries.

    params:

    :label_headers: a list of strings, each of which is the name of a column
    that contains a feature to be predicted (these features are 'murders',
    'murdPerPop', 'rapes', 'rapesPerPop', 'robberies', 'robbbPerPop', 'assaults',
    'assaultPerPop', 'burglaries', 'burglPerPop', 'larcenies', 'larcPerPop',
    'autoTheft', 'autoTheftPerPop', 'arsons', 'arsonsPerPop',
    'ViolentCrimesPerPop', and 'nonViolPerPop')
    :features_to_drop: a list of strings, each of which is the name of a column
    that contains a feature to be dropped due to its containing too many NaN
    entries (according to drop_nan_cols() and drop_nan_rows())

    returns: retained_label_headers, a list of strings, each of which contains
    the name of a feature to be predicted, but that hasn't been dropped due to
    its NaN content.
    """

    pass

def change_obj_to_int(inputs):
    """Converts the 'OtherPerCap' column, which is initially an object type,
    into int type, which it needs to be in order to be fed into your downstream
    learning algorithm.

    params:

    :inputs: a Pandas DataFrame that contains the inputs that will be taken in
    by the downstream learning algorithm.

    returns: inputs, a Pandas DataFrame for which the 'OtherPerCap' column
    is set to the type int.
    """
    for feature in inputs.columns: # Loop through all columns in the dataframe
        if inputs[feature].dtype == 'object': # Only apply for columns with categorical strings
            inputs[feature] = pd.Categorical(inputs[feature]).codes
    return inputs

def find_uncorrelated_cols(dataframe, drop_threshold=0.1,
                            analyzed_label="violentPerPop"):
    """Builds a correlations matrix from train_inputs and train_labels,
    and identifies any columns in train_inputs that is too weakly correlated with
    the analyzed_label (which will usually be 'ViolentCrimesPerPop', but which
    you can change if you wish).

    params:

    :train_inputs: a Pandas DataFrame that contains the training data.
    :train_labels: a Pandas DataFrame that contains the training labels,
    which include 'ViolentCrimesPerPop', among others.
    :drop_threshold: a float that provides the minimum correlation between
    each input feature and the analyzed_label in order for the feature not
    to be droped.
    :analyzed_label: a string providing the name of the label feature to be
    analyzed. By default, this should be 'ViolentCrimesPerPop'.

    returns: dropped_features, a list of strings, each of which indicates the
    name of a feature that will be dropped from our model because it's too
    uncorrelated to the feature indicated by analyzed_label
    """

    corr_matrix = dataframe.corr()
    orderd_corr_list = corr_matrix[analyzed_label]
    dropped_features = []
    for i in range(len(orderd_corr_list)):
        if abs(orderd_corr_list[i]) < 0.1:
            dropped_features.append(orderd_corr_list.index[i])

    return dropped_features


def drop_uncorrelated_cols(inputs, dropped_features):
    """Removes features that have been deemed insufficiently useful for our
    learning algorithm.

    params:

    :inputs: a Pandas DataFrame containing the data to be fed as input to our
    learning algorithm.
    :labels: a Pandas DataFrame containing the labels to be used by the
    learning algorithm.
    :dropped_features: a list of strings providing the names of features or labels
    that are to be dropped from the inputs DataFrame.

    returns: inputs and labels, both Pandas DataFrames respectively providing
    the inputs and labels to the learning algorithm, with the dropped_features
    removed.
    """
    inputs = inputs.drop(dropped_features,axis=1)
    return inputs

def normalize_dataset(dataset):
    """Normalizes all features in the dataset provided as input.

    params:

    :dataset: a Pandas DataFrame whose values are to be normalized by column.

    returns: dataset, a Pandas DataFrame whose values have been normalized to
    between 0 and 1. Hint: you'll probably want to use the MinMaxScaler()
    function, which has been imported for you at the top of this script.
    """
    scaler = MinMaxScaler()
    header = dataset.columns 
    dataset = scaler.fit_transform(dataset)
    dataset = pd.DataFrame(dataset)
    dataset.columns = header
    return dataset

def clean_and_normalize(dataset,usable_columns):
    """Removes columns from dataset that contain too many NaN values (by calling
    the function drop_nan_cols()), and subsequently, delete any remaining data
    points that contain NaNs (by calling the function drop_nan_rows()), and then
    normalize the data (by calling the function normalize_dataset()).

    params:

    :dataset: a Pandas DataFrame that contains the raw data as loaded from the
    CSV file.
    :usable_columns: a list of strings, each of which provides the name of a
    column that might potentially be used by the learning algorithm.

    returns: reduced_dataset, a Pandas DataFrame containing  the preprocessed
    dataset which does not contain any NaNs, and features_to_drop, a list of
    strings each of which is the name of a feature that is insufficiently
    correlated to the feature we want to analyze.
    """

    dataset = normalize_dataset(drop_nan_rows(drop_nan_cols(dataset,usable_columns)))
    return dataset

def generate_data(analyzed_label='ViolentCrimesPerPop',
                    all_headers_filepath='all_headers.json',
                    label_headers_filepath='label_headers.json',
                    dataset_filepath='data/CommViolPredUnnormalizedData.txt',
                    plot_correlations=False,
                    drop_threshold=0.1):
    """Generates inputs and labels for testing and training sets. You'll want
    to call all of the functions above to do so. Hint: be sure to follow
    the walkthrough if you need some additional detail on how to do this.

    params:

    :analyzed_label: a string providing the name of the feature to be analyzed.
    Should default to 'ViolentCrimesPerPop'.
    :plot_correlations: a boolean which, if True, will cause the correlations
    matrix to be plotted.
    :drop_threshold: a float providing the correlation threshold below which
    a feature will be dropped.

    returns: train_inputs, train_labels, test_inputs and test_labels, all of
    which are Pandas DataFrames, respectively containing the fully processed
    training inputs and labels, and testing inputs and labels.
    """

    names = load_attributeNames('attribute_names.csv')
    df = load_data('CommViolPredUnnormalizedData.txt',names[0])
    df = drop_nan_cols(df,names[0][5:])
    df = drop_nan_rows(df)
    inputs = change_obj_to_int(df)
    uncor_list = find_uncorrelated_cols(inputs)
    inputs = drop_uncorrelated_cols(inputs,uncor_list)
    inputs = normalize_dataset(inputs)
    train,test = split_train_and_test(inputs)

    train_data = train.drop(analyzed_label,axis=1)
    train_labels = train[analyzed_label]

    test_data = test.drop(analyzed_label,axis=1)
    test_labels = test[analyzed_label]

    return train_data,train_labels,test_data,test_labels

def plot_correlations(matrix):
    """Plots the correlations matrix obtained from the data provided.

    params:

    :matrix: the correlations matrix generated by pd.DataFrame.corr(),
    which provides the correlations between each input and the label
    of interest (in this case, 'ViolentCrimesPerPop').

    returns: nothing.
    """

    pass

def load_attributeNames(filepath):
    """Loads the attribute names from a file.

    params:

    :Filepath: the location and filename of the file containing the attribute names

    returns: a list containing the attributeNames
    """
    df = pd.read_csv(filepath,header=None)
    return df.values.T.tolist()

if __name__ == '__main__':
    names = load_attributeNames('attribute_names.csv')
    df = load_data('CommViolPredUnnormalizedData.txt',names[0])
    df = drop_nan_cols(df,names[0][5:])
    df = drop_nan_rows(df)
    inputs = change_obj_to_int(df)
    uncor_list = find_uncorrelated_cols(inputs)
    inputs = drop_uncorrelated_cols(inputs,uncor_list)
    inputs = normalize_dataset(inputs)
    train,test = split_train_and_test(inputs)
