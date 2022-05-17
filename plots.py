# This routine finds the MLE of the Virgo cluster center and covariance matrix under the multinomial normality assumption. It saves this data in a pickle file.


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import params
from utils import get_column, RA_to_hours, DEC_to_degrees
from clustering import clustering


def plot_RA (RA): # Plot the histogram of the RA distribution.
    plt.clf()
    plt.title("RA distribution", size=24)
    plt.xlabel('RA (hrs)', size=24)
    plt.ylabel('Number', size=24)
    plt.tick_params(labelsize=18)
    plt.hist(RA, bins=20, color='black',fill=False,linewidth=2,histtype='step')
    plt.gcf().set_size_inches(10,6)
    plt.gcf().savefig("plots/hist_RA.png", bbox_inches='tight', pad_inches=0.01, dpi=100, format='png')
    plt.gcf().savefig("plots/hist_RA.eps", bbox_inches='tight', pad_inches=0.01, dpi=100, format='eps')
    plt.close()
    
    
def plot_DEC (DEC): # Plot the histogram of the DEC distribution.
    plt.clf()
    plt.title("DEC distribution", size=24)
    plt.xlabel('DEC (deg)', size=24)
    plt.ylabel('Number', size=24)
    plt.tick_params(labelsize=18)
    plt.hist(DEC, bins=20, color='black',fill=False,linewidth=2,histtype='step')
    plt.gcf().set_size_inches(10,6)
    plt.gcf().savefig("plots/hist_DEC.png", bbox_inches='tight', pad_inches=0.01, dpi=100, format='png')
    plt.gcf().savefig("plots/hist_DEC.eps", bbox_inches='tight', pad_inches=0.01, dpi=100, format='eps')
    plt.close()
    
    
def plot_V_H (V_H): # Plot the histogram of the radial velocity distribution.
    plt.clf()
    plt.title("Radial velocity distribution", size=24)
    plt.xlabel('V (km/h)', size=24)
    plt.ylabel('Number', size=24)
    plt.tick_params(labelsize=18)
    plt.hist(V_H, bins=20, color='black',fill=False,linewidth=2,histtype='step')
    plt.gcf().set_size_inches(10,6)
    plt.gcf().savefig("plots/hist_V_H.png", bbox_inches='tight', pad_inches=0.01, dpi=100, format='png')
    plt.gcf().savefig("plots/hist_V_H.eps", bbox_inches='tight', pad_inches=0.01, dpi=100, format='eps')
    plt.close()

        
def plot_hist_membership (probs): # Plot the histogram of the Virgo membership probabilities.
    plt.clf()
    plt.title("Virgo membership probabilities", size=24)
    plt.xlabel('Probability', size=24)
    plt.ylabel('Number', size=24)
    plt.tick_params(labelsize=18)
    plt.hist(probs, bins=20, color='black',fill=False,linewidth=2,histtype='step')
    plt.gcf().set_size_inches(10,6)
    plt.gcf().savefig("plots/hist_membership.png", bbox_inches='tight', pad_inches=0.01, dpi=100, format='png')
    plt.gcf().savefig("plots/hist_membership.eps", bbox_inches='tight', pad_inches=0.01, dpi=100, format='eps')
    plt.close()
    
  
def plot_RA_DEC_map (RA, DEC, probs): # Plot the RA-DEC map of the Virgo membership probabilities.
    plt.clf()   
    N_RA, N_DEC = 400, 400
    overlap_factor = 20
    N_RA_ticks, N_DEC_ticks = 10, 10
    RA_step, DEC_step = (max(RA)-min(RA))/N_RA, (max(DEC)-min(DEC))/N_DEC
    RA_l = np.array([min(RA)+i*RA_step for i in range(N_RA)])
    RA_r = np.array([min(RA)+(i+1)*RA_step for i in range(N_RA)])
    DEC_l = np.array([min(DEC)+i*DEC_step for i in range(N_DEC)])
    DEC_r = np.array([min(DEC)+(i+1)*DEC_step for i in range(N_DEC)])
    df = np.zeros((N_RA,N_DEC))
    for i in range(N_RA):
        for j in range(N_DEC):
            df[i,j] = np.mean(probs[((RA>=RA_l[i]-overlap_factor*RA_step) & (RA<=RA_r[i]+overlap_factor*RA_step)) & ((DEC>=DEC_l[j]-overlap_factor*DEC_step) & (DEC<=DEC_r[j]+overlap_factor*DEC_step))])
    plt.figure(figsize=(N_RA,N_DEC))
    xticklabels, yticklabels = [], []
    for i in range(N_RA+1):
        if i % (N_RA/N_RA_ticks) == 0:
            xticklabels.append(round(min(RA)+RA_step*i,2))
        else:
            xticklabels.append('')
    for i in range(N_DEC+1):
        if i % (N_DEC/N_DEC_ticks) == 0:
            yticklabels.append(round(min(DEC)+DEC_step*i,2))
        else:
            yticklabels.append('')
    heatmap = sns.heatmap(df, vmin=0, vmax=1, cmap="afmhot_r", annot_kws={"size": 15}, xticklabels=xticklabels, yticklabels=yticklabels)
    heatmap.set_xticklabels(heatmap.get_xmajorticklabels(), fontsize = 15)
    heatmap.set_yticklabels(heatmap.get_ymajorticklabels(), fontsize = 15)
    heatmap.invert_yaxis()
    heatmap.set_title("Virgo membership probability map", fontdict={'fontsize':15}, pad=12)
    plt.xlabel(r'RA (hrs)', size=18)
    plt.ylabel(r'DEC (deg)', size=18)
    plt.gcf().set_size_inches(25.6, 14.4)
    plt.gcf().savefig("plots/RA_DEC_map.png", dpi=100)
    plt.gcf().savefig("plots/RA_DEC_map.eps", dpi=100)
    plt.close()
    
    
def plot_clusters(RA, DEC, clusters, k):
    plt.clf()
    plt.title("k = "+str(k)+" clusters", size=24)
    plt.xlabel("RA (hrs)", size=24)
    plt.ylabel("DEC (deg)", size=24)
    plt.tick_params(labelsize=18)
    plt.scatter(RA, DEC, c='grey', s=3)
    plt.scatter(RA[clusters[0]], DEC[clusters[0]], c='red', s=25)
    plt.scatter(RA[clusters[1]], DEC[clusters[1]], c='blue', s=25)
    plt.gcf().set_size_inches(25.6, 14.4)
    plt.gcf().savefig("plots/clusters_k_"+str(k)+".eps", dpi=100)
    plt.gcf().savefig("plots/clusters_k_"+str(k)+".png", dpi=100)
    plt.close()
        
        
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
    
    # Retrieve the normal distribution data.
    file = open(params.normal_pickle_filename, 'rb')
    data = pickle.load(file)
    probs = data[4]
    file.close()
    
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
    
    # Plot the data distributions.
    #plot_RA (RA)
    #plot_DEC (DEC)
    #plot_V_H (V_H)
    
    # Create the histogram of the membership probabilities, along with the RA/DEC map.
    #plot_hist_membership (probs)
    #plot_RA_DEC_map (RA, DEC, probs)
    
    # Calculate the similarity of the Virgo composition across multiple clustering parameter choices.
    cluster_list = []
    for min_cluster_size in params.minimum_cluster_sizes:
        clusters = clustering (RA_scaled, DEC_scaled, V_H_scaled, min_cluster_size=min_cluster_size)
        cluster_lengths = [len(c) for c in clusters]
        cluster_virgo_lengths = [len(np.intersect1d(c,prob_indeces)) for c in clusters]
        cluster_member_fractions = np.divide(cluster_virgo_lengths, cluster_lengths)
        cluster_ordering = np.argsort(-cluster_member_fractions)
        clusters = [clusters[i] for i in cluster_ordering]
        cluster_lengths = [len(c) for c in clusters]
        cluster_1, cluster_2 = clusters[0], clusters[1]
        cluster_list.append((cluster_1, cluster_2))
    
    # Plot the RA/DEC map of the Virgo vs non-Virgo clusters.
    for i, k in enumerate(params.minimum_cluster_sizes):
        plot_clusters(RA, DEC, cluster_list[i], k)