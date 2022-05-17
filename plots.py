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
    N_RA, N_DEC = 500, 500
    overlap_factor = 25
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
    plt.ylabel(r'DEC (days)', size=18)
    plt.gcf().set_size_inches(25.6, 14.4)
    plt.gcf().savefig("plots/RA_DEC_map.png", dpi=100)
    plt.gcf().savefig("plots/RA_DEC_map.eps", dpi=100)
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
    
    # Convert RA and DEC into the proper format.
    RA = np.array([RA_to_hours (a, b, c) for a, b, c in zip(ra_hh, ra_mm, ra_ss)])
    DEC = np.array([DEC_to_degrees (a, b, c) for a, b, c in zip(dec_dd, dec_mm, dec_ss)])
    
    # Retrieve the normal distribution data.
    file = open(params.normal_pickle_filename, 'rb')
    data = pickle.load(file)
    probs = data[4]
    file.close()
    
    # Create the histogram of the membership probabilities, along with the RA/DEC map.
    plot_hist_membership (probs)
    #plot_RA_DEC_map (RA, DEC, probs)
    