'''
Created on May 13, 2015

@author: Stefan Lattner
'''

import numpy as np

def peak_thresh(curve, n, k = 1):
    """
    Dynamic threshold implementation after Pearce, Muellensiefen, Wiggins 2010
    """
    if n > 2:
        w = np.linspace(0, 1, n-1)
        bound_mean = np.sqrt(sum([np.power(w[i]*curve[i] -
                                           np.average(curve[:n-1], weights=w),2)
                                  for i in range(n-1)]) / sum(w))

        slid_avg = sum([w[i]*curve[i] for i in range(n-1)]) / sum(w)
        thresh = k * bound_mean + slid_avg
    else:
        thresh = curve[n] + 1

    thresh = np.asarray(thresh)
    thresh[thresh == 0] = np.average(thresh)
    return thresh

