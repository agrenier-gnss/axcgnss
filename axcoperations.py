
import numpy as np

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

def correlation(array1, array2, axc_mult=None):

    N = len(array1)

    if axc_mult==None:
        axc_mult = mult

    corr = np.zeros(2*N-1)
    lags = np.zeros(2*N-1)
    idx = 0
    for lag in range(-N+1, N):
        correlation = 0
        for i in range(0, N):
            j = i + lag
            if j >= 0 and j < N:
                correlation += axc_mult(array1[j], array2[i])
        corr[idx] = correlation
        lags[idx] = lag
        idx += 1
    
    return corr, lags

# =============================================================================

import evoapproxlib as eal
import pandas

if __name__ == "__main__":

    # array1 = np.array([0,0,1,1,1,0,0])
    # array2 = np.array([0,1,1,1,0,0,0])
    
    # corr, lags = correlation(array1, array2)

    # print(corr, lags)

    # print(np.correlate(array1, array2, 'full'))

    # axc_bits = 2 # Lower bits approximated
    # bits = 4
    # max_value = 2**bits // 2
    # i_range = np.arange(-max_value, max_value-1, dtype=np.int16)
    # j_range = np.arange(-max_value, max_value-1, dtype=np.int16)
    # exact_results = np.zeros((2**bits, 2**bits))
    # axc_results = np.zeros((2**bits, 2**bits))
    # for i in i_range:
    #     for j in j_range:
    #         axc_results[i+max_value,j+max_value] = loadd(i, j, axc_bits)
    #         exact_results[i+max_value,j+max_value] = i + j
    
    # plt.figure()
    # plt.matshow(axc_results)
    # plt.colorbar()
    # plt.savefig('loadd.png')

    MAE_PERCENT = 18.75
    MAE = 805273600.0
    WCE_PERCENT = 75.0
    WCE = 3221094401.0
    WCRE_PERCENT = 100.0
    EP_PERCENT = 100.0
    MRE_PERCENT = 87.99
    MSE = 1.0407645e+18
    PDK45_PWR = 0.0003
    PDK45_AREA = 2.3
    PDK45_DELAY = 0.04
    
    kpis = [MAE_PERCENT, MAE, WCE_PERCENT, WCE, WCRE_PERCENT, EP_PERCENT, 
            MRE_PERCENT, MSE, PDK45_PWR, PDK45_AREA, PDK45_DELAY]
    df = pandas.DataFrame()
    for name, module in eal.multipliers['8x8_signed'].items():
        print(f"{name} | {module.MAE:>6} | {module.WCE:>6} | {module.MSE:>9} | {module.PDK45_PWR:>9}")
