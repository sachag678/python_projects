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

# This is module contains the function signatures for data exploration of the abalone
# dataset. The code is heavily commented to allow you to follow along easily.

# Please report any bugs you find using @yazabi, or email us at contact@yazabi.com.

import pandas as pd
import matplotlib.pyplot as plt
import data_preprocessing as dp
from sklearn.manifold import MDS, Isomap, TSNE
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from pandas.plotting import scatter_matrix
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import normalize
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV

def plot_raw_data(dataset):
    """Generates plot(s) to visualize the value of Rings as a function of
    every column of raw data in dataset. You may use whatever plotting format
    seems most appropriate (scatter plot, dot plot, histogram, etc.). The
    only critical thing is that any correlations between the Rings value and
    the other variables in dataset be made visible. Note: you may use different
    plot styles for different variables. For example, it probably makes sense
    to plot Rings vs Sex as a series of histograms, with Rings on the x-axis
    (this is because Sex is a categorical variable). For other variables,
    a scatter plot might make more sense.

    inputs:

    :dataset: a Pandas dataframe containing raw (unnormalized) data obtained
    from the data-containing CSV file.

    returns: nothing
    """
    num_attributes =  list(dataset.drop("Sex", axis=1))
    for att in num_attributes:
        if att != "Rings":
            dataset.plot(kind="scatter", x="Rings", y = att)
    plt.show()

    # dataset["Rings"].hist()
    # plt.show()


def correlated(dataset):
    corr_matrix = dataset.corr()

    print(corr_matrix["Rings"].sort_values(ascending=False))

    alpha = dataset.columns
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr_matrix)

    fig.colorbar(cax)

    ax.set_xticklabels(alpha,rotation=90)
    ax.set_yticklabels(alpha)

    plt.xticks(np.arange(0,len(alpha),1.0))
    plt.yticks(np.arange(0,len(alpha),1.0))

    plt.show()

def dimensionality_reduction(dataset, algorithm, sampled_frac=0.05):
    """Generates 2-D representations of the data provided in dataset,
    using the algorithm specified by the algorithm parameter. This script should
    save the plots it generates as JPEG files named 'mds.jpg', 'tsne.jpg' or
    'isomap.jpg', as appropriate (see below).

    params:

    :dataset: a pandas dataframe obtained by processing the raw data in the
    txt/csv file with the preprocess_dataset() function in the
    data_preprocessing.py module.

    :algorithm: a string providing the name of the algorithm to be used
    for dimensionality reduction. Can take on the values 'MDS', 'TSNE' or
    'isomap'. See http://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html,
    http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html and
    http://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html.

    :sampled_frac: a float indicating the fraction of samples in the total
    dataset that are to be used for dimensionality reduction and visualization.
    You'll notice that if you try to run TSNE, MDS or Isomap with all samples,
    it will take way too long to run - that's why we use such a small

    returns: nothing

    Hint 1: to save yourself some work, you can just use the split_train_and_test()
    function in the data_preprocessing.py module to obtain a reduced dataset.

    Hint 2: you'll probably want to color the points on your 2-D plots based on
    the category they belong to (i.e. their Rings value). You can do this by
    using plt.scatter(x_values,y_values,c=category_labels.astype(str)), where
    category_labels is a Pandas dataframe containing the Rings values for the
    reduced dataset that you're plotting.
    """
    num_samples = int(len(dataset.index)*sampled_frac)
    sample = dataset[:num_samples]
    category_labels = sample["Rings"]
    sample = sample.drop("Rings", axis =1)
    if algorithm =="PCA":
        dim_reducer = PCA(n_components=3)
    if algorithm == "MDS":
        dim_reducer = MDS(n_components = 3)
    if algorithm =="TSNE":
        dim_reducer = TSNE(n_components=3)
    if algorithm == "isomap":
        dim_reducer = Isomap(n_components=3)
    
    reduced_dataset = dim_reducer.fit_transform(sample)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')

    plt.cla()
    ax.scatter(reduced_dataset[:,0], reduced_dataset[:,1], reduced_dataset[:,2],c=category_labels.astype(str),cmap=plt.cm.spectral)

    ax.view_init(azim=30)
    plt.show()
    
    # plt.scatter(reduced_dataset[:,0], reduced_dataset[:,1], c=category_labels.astype(str))
    # plt.xlabel("z1")
    # plt.ylabel("z2")
    # plt.show()

def build_model(train_inputs,train_labels,model_params,model_mode='classification',
                    model_type='naive_bayes'):
    """Uses the training set to build a machine learning model with the parameters
    specified by the keywords model_params, model_mode and model_type. This function
    should allow the user to train a classifier using K nearest neighbors, a support
    vector machine, a decision tree or a naive Bayes algorithm, or a regressor
    using K nearest neighbors, a support vector machine or a decision tree (it's
    not possible to use naive Bayes for regression, so it can only be deployed
    for classification). For information on each algorithm, see the Python
    curriculum.

    params:

    :train_inputs: a Pandas dataframe obtained by passing appropriately
    preprocessed training data to the split_inputs_and_labels() function in the
    data_preprocessing.py module. Columns represent the features taken as
    input by our learning models.

    :train_labels: a Pandas dataframe likewise obtained from
    split_inputs_and_labels(), corresponding to the training set's Rings values.

    :model_params: a dictionary object containing the parameters to be used to
    train the model (e.g. for a KNeighborsClassifier, we might have
    model_params = {'n_neighbors': 5, 'leaf_size': 30, 'p': 2}).

    :model_mode: either 'classification' or 'regression'. Specifies whether the
    problem should be treated as a classification or regression problem.

    :model_type: 'naive_bayes', 'knn', 'svm' or 'decision_tree'. Indicates which
    model architecture is to be trained.

    returns: the trained model.
    """
    if model_mode == "classification":
        if model_type == "naive_bayes":
            model = GaussianNB()
        if model_type == "knn":
            model = KNeighborsClassifier(n_neighbors=50)
        if model_type == "svm":
            model = SVC(kernel='poly', degree =27, coef0 =1, C=5)
        if model_type == "decision_tree":
            model = DecisionTreeClassifier(min_samples_split=45,min_samples_leaf=45,criterion="gini")
            #model = RandomForestClassifier(n_estimators=500, n_jobs=-1)

    if model_mode == "regression":
        if model_type == "knn":
            model = KNeighborsRegressor()
        if model_type == "svm":
            model = SVR()
        if model_type == "decision_tree":
            model = DecisionTreeRegressor()


    model.fit(train_inputs, train_labels)
    # for name, score in zip(train_inputs.columns,model.feature_importances_):
    #     print(name, score)

    return model

def evaluate_model(model,test_inputs,test_labels,model_mode):
    """Evaluates the model passed as input using a test set. This function
    must: 1) print the model accuracy to the terminal (if in 'classification'
    mode) or print the mean standard error to the terminal (if in 'regression' mode)
    and 2) display (but not necessarily save) a plot of the confusion
    matrix obtained from the test set
    (see http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html).

    params:

    :model: the model to be evaluated.

    :test_inputs: a Pandas dataframe obtained by passing appropriately
    preprocessed testing data to the split_inputs_and_labels() function in the
    data_preprocessing.py module. Columns represent the features taken as
    input by our model.

    :test_labels: a Pandas dataframe likewise obtained from
    split_inputs_and_labels(), corresponding to the testing set's Rings values.

    :model_mode: either 'classification' or 'regression'. Specifies whether the
    problem should be treated as a classification or regression problem.

    returns: nothing.
    """

    if model_mode == "classification":
        y_pred = model.predict(test_inputs)
        print("Accuracy score: ", accuracy_score(test_labels, y_pred))
        #print("F1 score: ", f1_score(test_labels,y_pred, average='weighted'))

        conf_mx = confusion_matrix(test_labels, y_pred)
        #print(conf_mx)
        plt.matshow(conf_mx, cmap = plt.cm.jet)
        plt.show()

    if model_mode == "regression":
        y_pred = model.predict(test_inputs)
        print("Mean absolute error: ", mean_absolute_error(test_labels, y_pred))


if __name__ == '__main__':
    filepath = 'abalone.data.txt'
    #plot_raw_data(dp.load_dataset(filepath))
    #correlated(dp.preprocess_dataset(dp.load_dataset(filepath)))
    #dimensionality_reduction(dp.preprocess_dataset(dp.load_dataset(filepath)),"isomap")
    train_inputs, train_labels, test_inputs, test_labels = dp.generate_data()

    model_params = {}   
    model_mode = "classification"
    model_type = "svm"

    model = build_model(train_inputs, train_labels, model_params,model_mode, model_type)
    evaluate_model(model, test_inputs, test_labels,model_mode)
