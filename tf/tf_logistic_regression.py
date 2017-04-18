import tensorflow as tf

weighted_X = tf.matmul(X, weights)
weighted_X_with_bias = tf.add(weighted_X, bias)
prob = tf.nn.sigmoid(weighted_X_with_bias)