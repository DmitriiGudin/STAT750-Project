AUTHOR: DMITRII GUDIN (May 2022)


---


This is the list of scripts used in the course project of the Multivariate Analysis (STAT750) graduate course at University of Maryland in the Spring 2022 semester. The goal was to improve upon the results of the paper from 2020 by Selim et al. (https://www.scirp.org/journal/paperinformation.aspx?paperid=98148) and investigate Virgo cluster membership based on the dataset from Harvard (https://lweb.cfa.harvard.edu/~dfabricant/huchra/seminar/virgo/).

The following script files are included:


params.py: List of general parameters.

normal_center.py: Assumes the multinormality of the dataset (by RA, DEC and radial velocity), performs the MLE of the distribution parameters and saves them as a pickle file.

clustering.py: Performs the HDBSCAN clustering and saves the resulting clusters as a pickle file.

plots.py: Produces various plots used in the project paper. Run this after running all the scripts listed above.

analysis.py: Performs further analyses of the MLE and clustering results.

utils.py: Auxillary functions.