import numpy as np
import hdbscan
import pickle
import params
from utils import get_column, RA_to_hours, DEC_to_degrees
        
    
def clustering (RA_scaled, DEC_scaled, V_H_scaled, min_cluster_size=params.minimum_cluster_size): # Perform HDBSCAN. Return the list of clusters in descending order of their sizes.
    data = np.array([[a,b,c] for a, b, c in zip(RA_scaled, DEC_scaled, V_H_scaled)])
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=params.minimum_samples, cluster_selection_epsilon=params.cluster_selection_epsilon, cluster_selection_method='leaf', prediction_data=True)
    clusterer.fit(data)
    clusters = []
    N_clusters = clusterer.labels_.max()
    for i in range(N_clusters+1):
        clusters.append(list(np.where(clusterer.labels_==i)[0]))
    clusters = [c for c in clusters if len(c)>2]
    cluster_lengths = np.array([len(c) for c in clusters])
    clusters = [clusters[i] for i in np.argsort(-cluster_lengths)]
    return clusters
        
        
if __name__ == '__main__':

    # Retrieve the data from the file.
    ra_hh = get_column(1, str, params.virgo_filename, skip_header=2)
    ra_mm = get_column(2, str, params.virgo_filename, skip_header=2)
    ra_ss = get_column(3, str, params.virgo_filename, skip_header=2)
    dec_dd = get_column(4, str, params.virgo_filename, skip_header=2)
    dec_mm = get_column(5, str, params.virgo_filename, skip_header=2)
    dec_ss = get_column(6, str, params.virgo_filename, skip_header=2)
    V_H = get_column(8, float, params.virgo_filename, skip_header=2)
    V_H_err = get_column(9, float, params.virgo_filename, skip_header=2)
    
    # Apply the radial velocity cut.
    indeces = np.where(V_H>0)[0]
    ra_hh = ra_hh[indeces]
    ra_mm = ra_mm[indeces]
    ra_ss = ra_ss[indeces]
    dec_dd = dec_dd[indeces]
    dec_mm = dec_mm[indeces]
    dec_ss = dec_ss[indeces]
    V_H = V_H[indeces]
    V_H_err = V_H_err[indeces]
    
    # Convert RA and DEC into the proper format.
    RA = np.array([RA_to_hours (a, b, c) for a, b, c in zip(ra_hh, ra_mm, ra_ss)])
    DEC = np.array([DEC_to_degrees (a, b, c) for a, b, c in zip(dec_dd, dec_mm, dec_ss)])
    
    # Scale the clustering variables.
    RA_scaled = (RA-np.mean(RA))/np.std(RA)
    DEC_scaled = (DEC-np.mean(DEC))/np.std(DEC)
    V_H_scaled = (V_H-np.mean(V_H))/np.std(V_H)
    data_array = np.array([RA_scaled, DEC_scaled, V_H_scaled]).T
    
    # Perform HDBSCAN. Order the clusters in descending order of their sizes.
    clusters = clustering (RA_scaled, DEC_scaled, V_H_scaled)
    cluster_lengths = np.array([len(c) for c in clusters])
    print ("Clusters:", clusters)
    print ("Number of clusters:", len(clusters))
    print ("Sizes of clusters:", cluster_lengths)
    
    # Save the list of clusters to the *.pickle file.
    file = open(params.cluster_pickle_filename, 'wb')
    pickle.dump(clusters, file)
    file.close()
    