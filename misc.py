
import numpy as np
import scipy as sp

import sys
sys.path.append("/mnt/d/Projects/Navigation/MyCode/sydr")
from sydr.signal.gnsssignal import GenerateGPSGoldCode
from sydr.utils.constants import GPS_L1CA_CODE_FREQ

import axcoperations as axc

# =============================================================================

def getPowerdB(value_linear):
    return 10 * np.log10(value_linear)

# =============================================================================

def getPowerLinear(value_dB):
    return 10 ** (value_dB/10)

# =============================================================================

def generateGPSL1CA(prn=1, samplingFrequency=None):
    # Generate PRN code
    prn_code = GenerateGPSGoldCode(prn=prn, samplingFrequency=samplingFrequency)
    
    return prn_code

# =============================================================================

def addWhiteNoise(signal, sigma):

    noise = np.random.normal(0, scale=sigma, size=len(signal))
    
    #signal_out = signal + noise * sigma
    signal_out = signal + (noise - np.mean(noise))/np.std(noise)*sigma # From Simona's file
    #signal_out = signal + noise

    return signal_out

# =============================================================================

def getSigmaFromCN0(signal_power_dB, cn0_target_dB, signal_bw):
    
    snr_target_dB = cn0_target_dB - getPowerdB(signal_bw) # TODO Should we add the bandwidth of the sampling frequency as well? 

    noise_power = getPowerLinear(signal_power_dB) / getPowerLinear(snr_target_dB)
    sigma = np.sqrt(noise_power)

    return sigma

# =============================================================================

def quantize(signal, n_bits):

    # Old method
    # max_val = np.max(np.abs(signal))
    # num_levels = 2**n_bits 
    # bins = np.linspace(-max_val, max_val, num_levels)
    # centers = (bins[1:]+bins[:-1])/2
    # signal_quantized = np.digitize(signal, centers) - (num_levels // 2) 

    # Second method
    scale_factor = (2**n_bits // 2) / np.max(np.abs(signal)) 
    signal_quantized = np.round(signal * scale_factor - 0.5).astype(int)

    return signal_quantized, scale_factor


# =============================================================================

# def correlate(signal_1, signal_2, method='scipy', normalize=False):

#     # Normalization
#     # From https://stackoverflow.com/questions/53436231/normalized-cross-correlation-in-python
#     if normalize:
#         signal_1 = (signal_1 - np.mean(signal_1)) / (np.std(signal_1) * len(signal_1))
#         signal_2 = (signal_2 - np.mean(signal_2)) / (np.std(signal_2))

#     # Swith correlation method
#     match method:
#         case 'scipy':
#             corr = sp.signal.correlate(signal_1, signal_2, mode='full')
#             lags = sp.signal.correlation_lags(len(signal_1), len(signal_2), mode="full")
        
#         case 'axc':
#             corr, lags = axc.correlation(signal_1, signal_2)

#     return corr, lags

# =============================================================================

def getPostCorrelationSNR(correlation, samplingFrequency):

    samplesPerChip = round(samplingFrequency / GPS_L1CA_CODE_FREQ)

    # Get the max peak
    peak = np.max(correlation)**2
    maxPeakIdx = np.argmax(np.abs(correlation))

    # Get the mean outside the peak
    mask = np.ones(len(correlation), dtype=bool)
    mask[:maxPeakIdx-2*samplesPerChip] = False
    mask[maxPeakIdx+2*samplesPerChip:] = False
    noise = np.mean(np.abs(correlation[mask])**2)

    # Compute SNR
    snr = getPowerdB((peak - noise) / noise)

    return snr

