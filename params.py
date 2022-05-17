# Parameters relevant to this project.


import numpy as np


# File directories
virgo_filename = 'data/virgo.csv' # Location of the Virgo data file.
normal_pickle_filename = 'data/normal.pickle' # Location of the multivariate normal pickle file.
cluster_pickle_filename = 'data/cluster.pickle' # Location of the clustering pickle file.


# HDBSCAN parameters
minimum_cluster_size = 500 # Minimum number of data points per cluster.
minimum_samples = minimum_cluster_size
cluster_selection_epsilon = 0



# Normal-HDBSCAN comparison parameters 
minimum_cluster_sizes = [10, 100, 1000]
membership_confidence_level = 0.75