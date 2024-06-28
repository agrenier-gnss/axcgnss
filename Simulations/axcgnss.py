
#==================================================================================================
# Description: Library for simulations of Approximate Computing applied to GNSS
# Authors    : Antoine Grenier (TAU)
# Date       : 2024.05.14
#==================================================================================================
# MODULES
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import re 

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

def addWhiteNoise(signal, sigma, seed=None):

    if seed:
        np.random.seed(seed)

    noise = np.random.normal(0, scale=sigma, size=len(signal))
    
    #signal_out = signal + noise * sigma
    #signal_out = signal + noise
    signal_out = signal + (noise - np.mean(noise))/np.std(noise)*sigma # From Simona's file
    # AG: I believe this normalization is perform to have a noise power that is normalised (sigma_noise = 1) w.r.t to 
    #     the signal power. A explanation of this is given in the following reference 
    #     (GNSS Software Receivers, Kai Borre, 2023, p.28 (footnote))
    #     This is why we can assume 30 dB of gain during correlation, which explain the 30 dB in getSigmaFromCN0.

    return signal_out

# -------------------------------------------------------------------------------------------------

def getSigmaFromCN0(signal_power_dB, cn0_target_dB, signal_bw):

    # Based on (kai borre, 2023, p.28), and assuming noise power is normalised. 
    # correlation_gain = int(getPowerdB(GPS_L1CA_CODE_SIZE_BITS**2) - getPowerdB(GPS_L1CA_CODE_SIZE_BITS))

    correlation_gain = getPowerdB(signal_bw)
    
    snr_target_dB = cn0_target_dB - correlation_gain

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

#==================================================================================================
# ANALYSIS FUNCTIONS

def select_data(df, sampling_frequency=None, quantization=None, cn0=None):

    _df = df.copy()
    if sampling_frequency:
        _df = _df[_df['sampling_frequency'].eq(sampling_frequency)]
    
    if quantization: 
        _df = _df[_df['quantization'].eq(quantization)]

    if cn0: 
        _df = _df[_df['cn0_target_dB'].eq(cn0)]
    
    _df = _df.dropna(axis=1, how='all')

    return _df

# -------------------------------------------------------------------------------------------------

def plotSimulation(df, axc_mult):

    for index, row in df.iterrows():
        corr_lags = range(-row['delay_range'], row['delay_range']+1, row['delay_step'])
        plt.plot(corr_lags, row[f"axc_corr_{axc_mult}"])

    return 

# -------------------------------------------------------------------------------------------------

def plotCoherentCorrelation(df, axc_mult, num_integration):

    coherent_integration = np.zeros(401)
    idx = 0
    for index, row in df.iterrows():
        corr_lags = range(-row['delay_range'], row['delay_range']+1, row['delay_step'])
        plt.plot(corr_lags, row[f"axc_corr_{axc_mult}"])
        coherent_integration += row[f"axc_corr_{axc_mult}"]
        lags = corr_lags
        idx += 1
        if idx == num_integration:
            break
    plt.plot(lags, np.abs(coherent_integration))

    return 

# -------------------------------------------------------------------------------------------------

def plotPostCorrelationSNRPerFrequency(df_results, axc_mult_list, cn0_range, frequencies, bits, errors=None):

    idx_peak = 200
    
    plt.figure(figsize=(6,4))
    #plt.title(f"{axc_mult}")
    
    for axc_mult in axc_mult_list:
        snr_mean = []
        snr_std = []
        for cn0 in cn0_range:
            snr = []
            for freq in frequencies:
                df = select_data(df_results, int(freq), bits, cn0)
                for index, row in df.iterrows():
                    corr = np.abs(row[f"axc_corr_{axc_mult}"])
                    corr = removeCorrelationMean(corr, idx_peak, freq)
                    #idx_peak = np.argmax(corr)
                    snr.append(getPostCorrelationSNR(corr, row['sampling_frequency'], idx_peak))
            snr = np.array(snr)
            snr_mean.append(np.nanmean(snr))
            snr_std.append(np.nanstd(snr))
        snr_mean = np.array(snr_mean)
        snr_std = np.array(snr_std)
        
        label = re.sub('mul\d+s_', '', axc_mult)
        
        if label == '1KV6' or label == 'HG4':
            label = 'Exact'
        
        if errors == 'bar':
            plt.errorbar(cn0_range, snr_mean, snr_std, label=label, marker='o', capsize=3)
        elif errors == 'between':
            plt.plot(cn0_range, snr_mean, marker='o', label=label)
            plt.fill_between(cn0_range, snr_mean - snr_std, snr_mean + snr_std, alpha=0.2)
        else:
            plt.plot(cn0_range, snr_mean, marker='o', label=label)
    plt.grid()
    plt.xlabel("C/N0 [dB-Hz]")
    plt.ylabel("Post-correlation SNR [dB]")
    plt.xlim(cn0_range[0], cn0_range[-1])
    plt.legend(fontsize='small')
    return 

# -------------------------------------------------------------------------------------------------

def plotPostCorrelationSNRPerAxC(df_results, cn0_range, sampling_frequency, bits):

    idx_peak = 200

    if bits == 8:
        axc_mult_list = list(EAL_MULTIPLIERS_8BIT_SIGNED.keys())
    elif bits == 12:
        axc_mult_list = list(EAL_MULTIPLIERS_12BIT_SIGNED.keys())    
    elif bits == 16:
        axc_mult_list = list(EAL_MULTIPLIERS_16BIT_SIGNED.keys())

    snr = {'axc': axc_mult_list}
    for cn0 in cn0_range:
        snr_mean = []
        df = select_data(df_results, sampling_frequency, bits, cn0)
        
        for axc_mult in axc_mult_list:
            _snr = []
            for index, row in df.iterrows():
                corr = row[f'axc_corr_{axc_mult}']
                corr = removeCorrelationMean(corr, idx_peak, sampling_frequency)
                __snr = getPostCorrelationSNR(corr, sampling_frequency, idx_peak)

                # Get SNR exact
                corr = row[f'axc_corr_{axc_mult_list[0]}']
                corr = removeCorrelationMean(corr, idx_peak, sampling_frequency)
                snr_exact = getPostCorrelationSNR(corr, sampling_frequency, idx_peak)

                _snr.append( __snr - snr_exact)
            _snr = np.array(_snr)
            snr_mean.append(np.nanmean(_snr))
            snr[f'{cn0}'] = snr_mean
    snr = pd.DataFrame(snr).T
    snr = snr.rename(columns=snr.iloc[0]).iloc[1:]
    snr.plot(grid=True, legend=False)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    
    return

# -------------------------------------------------------------------------------------------------

def getPostCorrelationSNR(correlation, samplingFrequency, idxPeak):
    samplesPerChip = round(samplingFrequency / GPS_L1CA_CODE_FREQ)

    correlation = np.abs(correlation)

    # Get the max peak
    peak = correlation[idxPeak]**2

    # Get the mean outside the peak
    mask = np.ones(len(correlation), dtype=bool)
    mask[:idxPeak-2*samplesPerChip] = False
    mask[idxPeak+2*samplesPerChip:] = False
    noise = np.mean(np.abs(correlation[~mask])**2)

    # Compute SNR (from Simona) - Problem when noise > peak, lead to NaN values
    # snr = (peak - noise) / noise
    # snr = axg.getPowerdB(snr)
    if noise > peak:
        snr = 0
    else:
        snr = (peak - noise) / noise
        snr = getPowerdB(snr)
    
    # Compute SNR - does not remove the noise part in the peak, but at least can show negative SNR
    # if noise > peak:
    #     snr = 0
    # else:
    #     snr = getPowerdB(peak) - getPowerdB(noise)

    return snr

# -------------------------------------------------------------------------------------------------

def getCorrelationSeed(prn, sampling_frequency, cn0, correlator_delays, seed):

    signal_prn = GenerateGPSGoldCode(prn=prn)
    signal = UpsampleCode(signal_prn, sampling_frequency)
    # Generate the replicas
    replicas = GenerateDelayedReplicas(signal, correlator_delays)
    # Get noise from target CN0
    sigma_noise = getSigmaFromCN0(signal_power_dB=0, cn0_target_dB=cn0, signal_bw=int(sampling_frequency*2))
    # Noisy signal, create a pseudo-random seed
    signal_noisy = addWhiteNoise(signal, sigma=sigma_noise, seed=seed)
    
    corr, lags = PartialCorrelation(signal_noisy, replicas)

    return corr, lags

# -------------------------------------------------------------------------------------------------

def removeCorrelationMean(corr, idx_peak, sampling_frequency):
    
    samplesPerChip = round(sampling_frequency / GPS_L1CA_CODE_FREQ)
    mask = np.ones(len(corr), dtype=bool)
    mask[:idx_peak-2*samplesPerChip] = False
    mask[idx_peak+2*samplesPerChip:] = False
    mean = np.mean(corr[~mask])
    corr = np.abs(corr - mean)

    return corr