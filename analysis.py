import numpy as np
import pickle
import params
from utils import get_column, RA_to_hours, DEC_to_degrees        
from clustering import clustering
        
    
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
    
    # Retrieve the clustering and the probability data.
    file = open(params.normal_pickle_filename, 'rb')
    data = pickle.load(file)
    probs = data[4]
    file.close()
    file = open(params.cluster_pickle_filename, 'rb')
    clusters = pickle.load(file)
    file.close()
    
    # Get the indeces of galaxies in decreasing probability order:
    prob_indeces = np.argsort(-probs)
    prob_indeces = prob_indeces[int(params.membership_confidence_level*len(prob_indeces))+1:]
    
    # Calculate the similarity of the Virgo composition across multiple clustering parameter choices.
    for min_cluster_size in params.minimum_cluster_sizes:
        clusters = clustering (RA_scaled, DEC_scaled, V_H_scaled, min_cluster_size=min_cluster_size)
        cluster_lengths = [len(c) for c in clusters]
        cluster_virgo_lengths = [len(np.intersect1d(c,prob_indeces)) for c in clusters]
        cluster_member_fractions = np.divide(cluster_virgo_lengths, cluster_lengths)
        cluster_ordering = np.argsort(-cluster_member_fractions)
        print (cluster_ordering)
        clusters = [clusters[i] for i in cluster_ordering]
        cluster_lengths = [len(c) for c in clusters]
        cluster_1, cluster_2 = clusters[0], clusters[1]
        lens = (len(np.intersect1d(cluster_1,prob_indeces)), len(np.intersect1d(cluster_2,prob_indeces)))
        print ("Cluster lengths:", cluster_lengths)
        print ("Minimum cluster size:", min_cluster_size)
        print ("1st cluster size:", cluster_lengths[0])
        print ("Virgo members in the 1st cluster:", 100*lens[0]/cluster_lengths[0],"%")
        print ("2ndt cluster size:", cluster_lengths[1])
        print ("Virgo members in the 2st cluster:", 100*lens[1]/cluster_lengths[1],"%")
        
    # Estimate the number of true Virgo members according to the multinormal assumption.
    print ("Number of true Virgo members:", int(sum(probs)))