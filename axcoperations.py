
import numpy as np
import scipy as sp

# =============================================================================

def add(value1, value2):
    return value1 + value2

# =============================================================================

def mult(value1, value2):
    return value1 * value2

# =============================================================================

def loadd(value1, value2, N):
    """
    "Lower-part OR" adder 
    Approximate adders based on ignoring the lower bits. N is the number of lower bit approximated.
    @Author: Hans Jakob Damsgaard
    """
    mask = (1 << N) - 1
    return ((value1 & ~mask) + (value2 & ~mask)) + ((value1 & mask) | (value2 & mask))

# =============================================================================

state = 0
def lcadd(value1, value2, N):
    """
    "Lower-part Constant" adder 
    Approximate adders based on ignoring the lower bits. N is the number of lower bit approximated.
    A switch state is added to have symmetric errors in positive and negative values.
    @Author: Hans Jakob Damsgaard
    """
    global state
    #state = state ^ np.random.randint(0, 2)
    state = state ^ 1
    mask = (1 << N) - 1
    lsbs = (-state) & mask
    return ((value1 & ~mask) + (value2 & ~mask)) + lsbs

# =============================================================================

def FullCorrelation(array1, array2, axc_mult=None, axc_corr=0):

    N = len(array1)

    if axc_mult==None:
        corr = sp.signal.correlate(array1, array2, mode='full')
        lags = sp.signal.correlation_lags(len(array1), len(array2), mode="full")
        return corr, lags

    corr = np.zeros(2*N-1)
    lags = np.zeros(2*N-1)
    idx = 0
    for lag in range(-N+1, N):
        correlation = 0
        for i in range(0, N):
            j = i + lag
            if j >= 0 and j < N:
                correlation += axc_mult(array1[j], array2[i]) + axc_corr
        corr[idx] = correlation
        lags[idx] = lag
        idx += 1
    
    return corr, lags


def PartialCorrelation(array1, replicas, axc_mult=None, axc_corr=0):

    N = len(array1)

    corr = np.zeros(len(replicas.keys()))
    lags = np.zeros(len(replicas.keys()))
    idx = 0
    for delay, array2 in replicas.items():
        if axc_mult==None:
            corr[idx] = np.sum(array1 * array2)
            lags[idx] = delay
        else:
            correlation = 0
            for i in range(0, N):
                correlation += axc_mult(array1[i], array2[i]) + axc_corr
            corr[idx] = correlation
            lags[idx] = delay
        idx += 1

    return corr, lags

