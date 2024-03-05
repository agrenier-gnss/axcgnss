
import numpy as np

def add(value1, value2):
    return value1 + value2

def mult(value1, value2):
    return value1 * value2

def sum(array1):
    result = 0
    for value in array1:
        result += value
    return result


def correlation(array1, array2):

    N = len(array1)

    corr = np.zeros(2*N-1)
    lags = np.zeros(2*N-1)
    idx = 0
    for lag in range(-N+1, N):
        correlation = 0
        for i in range(0, N):
            j = i + lag
            if j >= 0 and j < N:
                correlation += array1[j] * array2[i]
        corr[idx] = correlation
        lags[idx] = lag
        idx += 1
    
    return corr, lags


if __name__ == "__main__":

    array1 = np.array([0,0,1,1,1,0,0])
    array2 = np.array([0,1,1,1,0,0,0])
    
    corr, lags = correlation(array1, array2)

    print(corr, lags)

    print(np.correlate(array1, array2, 'full'))

