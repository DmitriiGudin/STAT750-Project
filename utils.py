# Utility file containing commonly used functions.


import numpy as np


def get_column(N, Type, filename, skip_header=0): # Retrieve the Nth column of the specified *.csv file.
    return np.genfromtxt(filename, delimiter=';', dtype=Type, skip_header=skip_header, usecols=N, comments=None)


def RA_to_hours (hh, mm, ss): # Convert the RA value from hh:mm:ss to numerical.
    return float(hh)+float(mm)/60+float(ss)/(60**2)


def DEC_to_degrees (dd, mm, ss): # Convert the DEC value from ddd:mm:ss to numerical.
    val = float(dd[1:]) + float(mm)/60 + float(ss)/(60**2)
    if dd[0]=='+':
        return val
    else:
        return -val