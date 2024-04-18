
import numpy as np
import matplotlib.pyplot as plt

COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

# =============================================================================

def plotSignals(signals, sampling_frequency, code_frequency, title):

    fig = plt.figure(figsize=(12, 2.5*len(signals)), layout="constrained")
    spec = fig.add_gridspec(len(signals), 3)
    fig.suptitle(title, size='x-large', fontweight='bold')

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    lim_psd = code_frequency*2
    
    axs = []
    idx = 0
    for name, signal in signals.items():
        ax_plot = fig.add_subplot(spec[idx, 0:2])
        ax_plot.plot(signal, color=COLORS[idx])
        ax_plot.set_title(name)
        ax_plot.set_ylabel('Amplitude [unitless]')
        ax_plot.set_xlabel('Samples')

        ax_psd = fig.add_subplot(spec[idx, 2:])
        ax_psd.psd(signal, Fs=sampling_frequency, sides='twosided', color=COLORS[idx])
        ax_psd.set_ylabel("PSD [dB/Hz]")
        ax_psd.set_xlim(-lim_psd, lim_psd)

        axs.append(ax_plot)
        axs.append(ax_psd)
        idx += 1

    for ax in axs:
        ax.grid(visible=True)
    
    plt.show()

    return

# =============================================================================

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