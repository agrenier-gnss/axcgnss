import sys
sys.path.append("/mnt/c/Users/vmangr/Documents/Code/sydr")
from sydr.utils.constants import GPS_L1CA_CODE_FREQ 
from sydr.signal.gnsssignal import GenerateGPSGoldCode, UpsampleCode

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import multiprocessing

from plot import plotCorrelation, plotSignals
from misc import *
from axcoperations import loadd


import evoapproxlib as eal
from axcoperations import correlate, mult

# =================================================================================================
# Signal parameters
prn = 1
signal_prn = GenerateGPSGoldCode(prn=prn)

# =================================================================================================

def worker(sigma_noise, quantization_bits, custom_multiplier):
    
    signal = UpsampleCode(signal_prn, sf)

    # Noisy signal
    signal_noisy = addWhiteNoise(signal, sigma=sigma_noise)

    # Quantized signal 
    signal_quantized, scale_factor = quantize(signal_noisy, quantization_bits)

    # Perform correlations
    signal_noisy_corr, signal_noisy_lags         = correlate(signal, signal_noisy)
    signal_quantized_corr, signal_quantized_lags = correlate(signal, signal_quantized)
    

    # Save results
    results = {
        "sigma_noise"          : sigma_noise,
        "signal_noisy"         : signal_noisy,
        "signal_quantized"     : signal_quantized,
        "scale_factor"         : scale_factor,
        "signal_noisy_corr"    : signal_noisy_corr,
        "signal_noisy_lags"    : signal_noisy_lags,
        "signal_quantized_corr": signal_quantized_corr, 
        "signal_quantized_lags": signal_quantized_lags,
    }

    for name, _mult in custom_multipliers:
        signal_axc_corr, signal_axc_lags = correlate(signal, signal_quantized, axc_mult=_mult)
        results[f"signal_axc_corr_{name}"] = signal_axc_corr
        results[f"signal_axc_lags_{name}"] = signal_axc_lags

    return results

if __name__ == "__main__":

    # Parameters
    # sf_list = [2, 4, 8, 16, 20] * GPS_L1CA_CODE_FREQ
    # cn0_list = range(30, 60, 5)
    # bits_list = [8, 16]
    # nb_run = 100 # Number Monte Carlo run per subset of parameter
    # signal_bw = GPS_L1CA_CODE_FREQ # Bandwidth of the signal for CN0 to SNR computation
    
    sf_list = [10] * GPS_L1CA_CODE_FREQ
    cn0_list = [60]
    bits_list = [8, 16]
    nb_run = 1 # Number Monte Carlo run per subset of parameter
    signal_bw = GPS_L1CA_CODE_FREQ # Bandwidth of the signal for CN0 to SNR computation

    # Select custom multipliers
    custom_multipliers = {}
    for name, module in eal.multipliers['8x8_signed'].items():
        custom_multipliers[name] = getattr(module, 'calc')
    for name, module in eal.multipliers['16x16_signed'].items():
        custom_multipliers[name] = getattr(module, 'calc')
    
    # Multiprocessing
    num_processes = 4  # CPU cores
    pool = multiprocessing.Pool(processes=num_processes)

    # Simulations
    args = []
    for sf in sf_list:
        for bits in bits_list:
            _custom_mult = {k: v for k, v in custom_multipliers.items() if k.startswith(f'{bits}')}
            for cn0 in cn0_list:
                for _ in nb_run:
                    sigma_noise = getSigmaFromCN0(signal_power_dB=0, cn0_target_dB=cn0, signal_bw=signal_bw)
                    args.append(sigma_noise, sf, bits, _custom_mult)

    results = pool.starmap(worker, args)

    pool.close()
    pool.join()
    print("Done")