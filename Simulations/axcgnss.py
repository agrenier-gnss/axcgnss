
#==================================================================================================
# Description: Library for simulations of Approximate Computing applied to GNSS
# Authors    : Antoine Grenier (TAU)
# Date       : 2024.05.14
#==================================================================================================
# MODULES
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import evoapproxlib as eal
import ca

#==================================================================================================
# CONSTANTS
GPS_L1CA_CODE_SIZE_BITS = 1023     # Number bit per C/A code    
GPS_L1CA_CODE_FREQ = 1.023e6       # [Hz] C/A code frequency 

EAL_MULTIPLIERS_8BIT_SIGNED = {
    'mul8s_1KV6' : eal.mul8s_1KV6.calc,
    'mul8s_1KVM' : eal.mul8s_1KVM.calc,
    'mul8s_1KXF' : eal.mul8s_1KXF.calc,
    'mul8s_1L12' : eal.mul8s_1L12.calc,
    #'mul8s_1L2J' : eal.mul8s_1L2J.calc,
    'mul8s_1KV8' : eal.mul8s_1KV8.calc,
    'mul8s_1KV9' : eal.mul8s_1KV9.calc,
    'mul8s_1KVP' : eal.mul8s_1KVP.calc,
    #'mul8s_1L2L' : eal.mul8s_1L2L.calc,
    #'mul8s_1L2N' : eal.mul8s_1L2N.calc,
    'mul8s_1KVQ' : eal.mul8s_1KVQ.calc,
    'mul8s_1KX5' : eal.mul8s_1KX5.calc,
    'mul8s_1KVA' : eal.mul8s_1KVA.calc
}

EAL_MULTIPLIERS_12BIT_SIGNED = {
    'mul12s_2PP' : eal.mul12s_2PP.calc,
    'mul12s_2PQ' : eal.mul12s_2PQ.calc,
    'mul12s_2PR' : eal.mul12s_2PR.calc,
    'mul12s_2PS' : eal.mul12s_2PS.calc,
    'mul12s_2PT' : eal.mul12s_2PT.calc,
    'mul12s_2QD' : eal.mul12s_2QD.calc,
    'mul12s_2QE' : eal.mul12s_2QE.calc,
    'mul12s_2QH' : eal.mul12s_2QH.calc,
    'mul12s_34K' : eal.mul12s_34K.calc,
    'mul12s_2R5' : eal.mul12s_2R5.calc,
    'mul12s_2RP' : eal.mul12s_2RP.calc,
    'mul12s_34M' : eal.mul12s_34M.calc,
    'mul12s_34P' : eal.mul12s_34P.calc,
    'mul12s_2TE' : eal.mul12s_2TE.calc
}

EAL_MULTIPLIERS_16BIT_SIGNED = {
    'mul16s_HG4' : eal.mul16s_HG4.calc,
    'mul16s_GQU' : eal.mul16s_GQU.calc,
    'mul16s_GQV' : eal.mul16s_GQV.calc,
    'mul16s_HF7' : eal.mul16s_HF7.calc,
    'mul16s_GRU' : eal.mul16s_GRU.calc,
    'mul16s_GSM' : eal.mul16s_GSM.calc,
    'mul16s_HG8' : eal.mul16s_HG8.calc,
    'mul16s_HFB' : eal.mul16s_HFB.calc,
    'mul16s_GV3' : eal.mul16s_GV3.calc
}

#==================================================================================================
# GNSS FUNCTIONS

def GenerateGPSGoldCode(prn, samplingFrequency=None):
    """
    Return satellite PRN code. If a sampling frequency is provided, the
    code returned code will be upsampled to the frequency. 

    Args:
        samplingFrequency (float, optional): Sampling frequency of the code

    Returns:
        code (ndarray): array of PRN code.

    Raises:
        ValueError: If self.signalType not recognised. 
    """

    # Generate gold code
    code = ca.code(prn, 0, 0, 1, GPS_L1CA_CODE_SIZE_BITS)

    # Raise to samping frequency
    if samplingFrequency:
        code = UpsampleCode(code, samplingFrequency)

    return code

# -------------------------------------------------------------------------------------------------

def UpsampleCode(code, samplingFrequency:float):
    """
    Return upsampled version of code given a sampling frequency.

    Args:
        code (ndarray): Code to upsample.
        samplingFrequency (float): Sampling frequency of the code

    Returns:
        codeUpsampled (ndarray): Code upsampled.
    """
    ts = 1/samplingFrequency     # Sampling period
    tc = 1/GPS_L1CA_CODE_FREQ    # C/A code period
    
    # Number of points per code 
    samples_per_code = getSamplesPerCode(samplingFrequency)
    
    # Index with the sampling frequencies
    idx = np.trunc(ts * np.array(range(samples_per_code)) / tc).astype(int)
    
    # Upsample the original code
    codeUpsampled = code[idx]

    return codeUpsampled

# -------------------------------------------------------------------------------------------------

def getSamplesPerCode(samplingFrequency:float):
    """
    Return the number of samples per code given a sampling frequency.

    Args:
        samplingFrequency (float): Sampling frequency of the code.

    """
    return round(samplingFrequency / (GPS_L1CA_CODE_FREQ / GPS_L1CA_CODE_SIZE_BITS))


# -------------------------------------------------------------------------------------------------

def GenerateDelayedReplicas(signal, correlator_delays):

    replicas = {}
    for delay in correlator_delays: 
        replicas[delay] = np.roll(signal, delay)

    return replicas

# -------------------------------------------------------------------------------------------------

def addWhiteNoise(signal, sigma):

    noise = np.random.normal(0, scale=sigma, size=len(signal))
    
    #signal_out = signal + noise * sigma
    signal_out = signal + (noise - np.mean(noise))/np.std(noise)*sigma # From Simona's file
    #signal_out = signal + noise

    return signal_out

# -------------------------------------------------------------------------------------------------

def getSigmaFromCN0(signal_power_dB, cn0_target_dB, signal_bw):
    
    snr_target_dB = cn0_target_dB - getPowerdB(signal_bw) # TODO Should we add the bandwidth of the sampling frequency as well? 

    noise_power = getPowerLinear(signal_power_dB) / getPowerLinear(snr_target_dB)
    sigma = np.sqrt(noise_power)

    return sigma

# -------------------------------------------------------------------------------------------------

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


#==================================================================================================
# AXC FUNCTIONS 

def exact_add(value1, value2):
    return value1 + value2

# -------------------------------------------------------------------------------------------------

def exact_mult(value1, value2):
    return value1 * value2

# -------------------------------------------------------------------------------------------------

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

# -------------------------------------------------------------------------------------------------

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


#==================================================================================================
# PLOTTING FUNCTIONS

def plotCorrelation(correlations, sampling_frequency, code_frequency, title, normalised=True):

    fig = plt.figure(figsize=(12, 4), layout="constrained")
    spec = fig.add_gridspec(1, 3)
    fig.suptitle(title, size='x-large', fontweight='bold')

    samplesPerChip = round(sampling_frequency / code_frequency)
    lim_corr_large = 100
    lim_corr_small = 5

    idx = 0 
    ax_large = fig.add_subplot(spec[idx:, 0:2])
    ax_small = fig.add_subplot(spec[idx:, 2:])
    for name, corr in correlations.items():
        corr_values, corr_lags = corr
        #ax_large.plot(corr_lags/samplesPerChip, corr_values / np.max(corr_values), color=COLORS[idx], label=name)
        ax_large.plot(corr_lags/samplesPerChip, corr_values, color=COLORS[idx], label=name)
        #ax_large.set_xlim(-lim_corr_large, lim_corr_large)
        ax_large.set_title("Correlation")
        ax_large.set_xlabel('Chips')

        #ax_small.plot(corr_lags/samplesPerChip, corr_values / np.max(corr_values), color=COLORS[idx], label=name)
        ax_small.plot(corr_lags[4950:5050]/samplesPerChip, corr_values[4950:5050], color=COLORS[idx], label=name)
        #ax_small.set_xlim(-lim_corr_small, lim_corr_small)
        ax_small.set_xlabel('Chips')

        idx += 1

    axs = [ax_large, ax_small]
    for ax in axs:
        ax.grid(visible=True)
        ax.legend()
    
    plt.show()

    return

#==================================================================================================
# MISC FUNCTIONS

def getPowerdB(value_linear):
    return 10 * np.log10(value_linear)

# -------------------------------------------------------------------------------------------------

def getPowerLinear(value_dB):
    return 10 ** (value_dB/10)

# -------------------------------------------------------------------------------------------------

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

