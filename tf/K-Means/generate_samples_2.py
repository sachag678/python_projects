import tensorflow as tf
import numpy as np

from tf_k_means import create_samples, plot_clusters, choose_random_centroids, assign_to_nearest,update_centroids

n_features = 2
n_clusters = 3 #num_classes
n_samples_per_cluster = 500
seed = 700
embiggen_factor = 70

np.random.seed(seed)

centroids, samples = create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)

#is there a cleaner way of setting up the iteration through the
model = tf.global_variables_initializer()
with tf.Session() as session:
	sample_values = session.run(samples)
	#get initial
	initial_centroids = choose_random_centroids(samples, n_clusters, seed)
	centroid_values = session.run(initial_centroids)
	#get nearest
	nearest_indices = assign_to_nearest(samples, centroid_values)
	nearest_indices_values = session.run(nearest_indices)
	#update centroids
	updated_centroids = update_centroids(samples, nearest_indices_values, n_clusters)
	centroid_values = session.run(updated_centroids)
	
	for i in range(1000):
	    nearest_indices_values = session.run(nearest_indices) 
	    centroid_values = session.run(updated_centroids)
	
	plot_clusters(sample_values, centroid_values, n_samples_per_cluster)