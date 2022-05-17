# This routine finds the MLE of the Virgo cluster center and covariance matrix under the multinomial normality assumption. Also calculates the Mahalanobis distances and membership probabilities of all galaxies. It saves this data to a *.pickle file.


import numpy as np
from scipy.stats import multivariate_normal, chi2
from scipy.optimize import minimize
import pickle
import params
from utils import get_column, RA_to_hours, DEC_to_degrees
    
    
def clustering (RA_scaled, DEC_scaled, V_H_scaled):
    data = np.array([[a,b,c] for a, b, c in zip(RA_scaled, DEC_scaled, V_H_scaled)])
    clusterer = hdbscan.HDBSCAN(min_cluster_size=minimum_cluster_size, min_samples=minimum_samples, cluster_selection_epsilon=cluster_selection_epsilon, cluster_selection_method='leaf', prediction_data=True)
    clusterer.fit(data)
    groups = []
    N_groups = clusterer.labels_.max()
    for i in range(N_groups+1):
        groups.append(list(np.where(clusterer.labels_==i)[0]))
    return groups
        
        
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
    
    # Convert RA and DEC into the proper format.
    RA = np.array([RA_to_hours (a, b, c) for a, b, c in zip(ra_hh, ra_mm, ra_ss)])
    DEC = np.array([DEC_to_degrees (a, b, c) for a, b, c in zip(dec_dd, dec_mm, dec_ss)])
    
    # Scale the clustering variables.
    RA_scaled = (RA-np.mean(RA))/np.std(RA)
    DEC_scaled = (DEC-np.mean(DEC))/np.std(DEC)
    V_H_scaled = (V_H-np.mean(V_H))/np.std(V_H)
    data_array = np.array([RA_scaled, DEC_scaled, V_H_scaled]).T
    
    # Define the negative log-likelihood function for our dataset. x is an array in which the first n elements are the mean vector and the subsequent n^2 elements are the covariance matrix.
    def neg_log_likelihood (x):
        return -sum([multivariate_normal.logpdf(data_array[i], x[0:3], x[3:].reshape((3,3))) for i in range(len(data_array))])
        
    # Use the BFGS method to maximize the log-likelihood. Print out the results of the maximization.
    length = len(data_array[0])
    center = minimize(neg_log_likelihood, np.array(list(np.zeros(length))+list(np.identity(length).flatten())), method='L-BFGS-B', options={'maxiter': 20, 'gtol': 1e-6, 'disp': True}).x
    print ("")
    print ("")
    print ("Center coordinates:")
    print ("RA:", center[0]*np.std(RA)+np.mean(RA))
    print ("DEC:", center[1]*np.std(DEC)+np.mean(DEC))
    print ("V_H:", center[2]*np.std(V_H)+np.mean(V_H), "km/s")
    mu = center[0:3]
    sigma = center[3:].reshape((3,3))
    print ("Mean vector:", mu)
    print ("Covariance matrix:", sigma)
    
    # Calculate Mahalanobis distances of all datapoints to the center. Calculate the corresponding membership probabilities.
    m_dist = np.array([np.dot((x-mu).T,np.linalg.inv(sigma)) for x in data_array])
    m_dist = np.array([np.dot(d,x-mu) for d,x in zip(m_dist,data_array)])
    print ("Mahalanobis distances:", m_dist)
    probs = np.array([1-chi2.cdf(d, 3) for d in m_dist])
    print ("Probabilities:", probs)
    
    # Save these results to the *.pickle file.
    file = open(params.normal_pickle_filename, 'wb')
    pickle_data = []
    pickle_data.append('mu, sigma, m_dist, probs')
    pickle_data.append(mu)
    pickle_data.append(sigma)
    pickle_data.append(m_dist)
    pickle_data.append(probs)
    pickle.dump(pickle_data, file)
    file.close()