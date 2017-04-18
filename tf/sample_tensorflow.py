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

# This is example code showing how to implement object-oriented TensorFlow.
# Note that normally each class would be in its own module (.py file). The code
# is heavily commented to allow you to follow along easily.

# Please report any bugs you find using @yazabi, or email us at contact@yazabi.com.

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

SAMPLE_DIM = 5
SAMPLES_PER_BATCH = 10
NUM_BATCHES = 100
LEARNING_RATE = 0.1

class DataGenerator:

    def __init__(self, sample_dim = SAMPLE_DIM):
        """Initialize an instance of the DataGenerator class.

        sample_dim:the dimensionality of the example inputs
        """

        self.sample_dim = sample_dim

        # We define the "true weights" of the system. These are the
        # values that nature has decided the weights should actually
        # have, and therefore, the values that our learning algorithm
        # will have to determine.
        self.true_weights = np.random.randint(10, size = self.sample_dim)

    def generate_data(self,num_samples):
        """Returns samples of input data and their labels.

        num_samples: the number of examples (and labels) to be returned
        """

        # samples will be a matrix whose rows correspond to different samples
        # that will be fed into the neural net
        self.samples = np.random.rand(num_samples, self.sample_dim)

        # labels calculated by multiplying each dimension of each sample by
        # the corresponding "true weight"
        labels = np.dot(self.samples, self.true_weights)

        # define a noise term to be added to the labels we're adding the
        # noise just to make things a little nontrivial for the learning
        # algorithm later on)
        noise = np.random.rand(num_samples)
        
        # define the noisy labels
        labels = labels + noise

        # formatting for feed-in to neural net
        self.labels = [[label] for label in labels]

class GraphConstructor:

    def __init__(self, sample_dim=SAMPLE_DIM):
        """Initializes an instance of the GraphConstructor class.

        sample_dim: the dimensionality of the input data.
        """

        self.sample_dim = sample_dim

    def build_graph(self,learning_rate=LEARNING_RATE):
        """Builds the computational graph that will be evaluated during
        training.
        """

        # defining our placeholders. samples is the set of sample examples
        # that constitutes this batch, and true_labels are their labels.
        # now, why are we defining samples and true_labels as properties of
        # this class? It's because we want the pointers to these values
        # in order to be able to feed in data during training.
        self.samples = tf.placeholder(tf.float32, [None, self.sample_dim])
        self.true_labels = tf.placeholder(tf.float32, [None, 1])

        # using tf.get_variable rather than tf.Variable, because, well, it's
        # just plain better. Initializing with Xavier initialization
        self.W = tf.get_variable('weights',[self.sample_dim,self.sample_dim],initializer=tf.contrib.layers.xavier_initializer())

        self.guessed_labels = tf.matmul(self.samples,self.W)

        # just using a simple mean square error loss function
        self.loss = tf.reduce_mean(tf.square(self.guessed_labels - self.true_labels))

        # define the training op (simple gradient descent, nothing fancy)
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)

        # NOTE: in versions of TensorFlow that predate v1.0, this command
        # changes to tf.initialize_all_variables()
        self.model = tf.global_variables_initializer()

class Learn:

    def __init__(self, num_batches=NUM_BATCHES,samples_per_batch=SAMPLES_PER_BATCH):
        """Initializes an instance of the Algorithm class.

        num_batches: the number of training batches.

        samples_per_batch: the number of samples in each training batch.
        """

        # define a few key training parameters
        self.num_batches = num_batches
        self.samples_per_batch = samples_per_batch

        # build the computational graph that will be used to predict
        # the true weights
        self.graph = GraphConstructor()
        self.graph.build_graph()

        self.data = DataGenerator()

    def run_training(self):
        """Runs the actual training loops and records the values of the
        loss function with each batch.
        """

        # just for convenience, define graph and data without the call to self
        graph = self.graph
        data = self.data

        # create a TensorFlow session
        with tf.Session() as session:

            # initialize the model's global variables
            session.run(graph.model)

            # training losses will hold the values of the training loss
            # after each training batch
            training_losses = []

            # run the training loop
            for batch in np.arange(self.num_batches):

                # generate a new dataset to be fed in at each training iteration
                data.generate_data(self.samples_per_batch)
                
                # define the feed_dict that will be fed in at each training
                # iteration
                feed_dict = {graph.samples: data.samples,
                             graph.true_labels: data.labels}

                # run the training operation and recover the resulting loss value
                _, loss_value = session.run([graph.train_op,graph.loss],feed_dict=feed_dict)

                # record the value of the loss
                training_losses += [loss_value]

        self.training_losses = training_losses

    def plot_training_loss(self):
        """Plots the training loss as a function of batch number.
        """

        # plotting the training loss as a function of batch number
        plt.plot(self.training_losses)
        plt.show()
