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
initial_centroids = choose_random_centroids(samples, n_clusters, seed)
nearest_indices = assign_to_nearest(samples, initial_centroids)
updated_centroids = update_centroids(samples, nearest_indices, n_clusters)

#is there a cleaner way of setting up the iteration through the
model = tf.global_variables_initializer()
with tf.Session() as session:
	sample_values = session.run(samples)
	centroid_values = session.run(initial_centroids)
	updated_centroid_values = session.run(updated_centroids)
	#redefinition of nearest to use the updated values
	nearest_indices = assign_to_nearest(samples, updated_centroid_values)
	
	for i in range(1000):
	    nearest_indices_values = session.run(nearest_indices)
	    #redefintion of update_centroids to use nearest_indeices_values
	    updated_centroids = update_centroids(samples, nearest_indices_values, n_clusters)   
	    updated_centroid_values = session.run(updated_centroids)
	    #print(updated_centroid_values)
	
	plot_clusters(sample_values, updated_centroid_values, n_samples_per_cluster)