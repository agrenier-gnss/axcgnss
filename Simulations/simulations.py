import numpy as np
import matplotlib.pyplot as plt
import pickle
import multiprocessing
import time

import axcgnss as axg

# ======================================================================================================================

def worker(run, prn, signal, replicas, cn0_target_dB, sampling_frequency, quantization_bits, custom_multipliers):

    signal_bw = axg.GPS_L1CA_CODE_FREQ # Bandwidth of the signal for CN0 to SNR computation

    # Get noise from target CN0
    sigma_noise = axg.getSigmaFromCN0(signal_power_dB=0, cn0_target_dB=cn0_target_dB, signal_bw=signal_bw)

    # Noisy signal
    signal_noisy = axg.addWhiteNoise(signal, sigma=sigma_noise)

    # Quantized signal 
    signal_quantized, scale_factor = axg.quantize(signal_noisy, quantization_bits)
    
    # Save results
    results = {
        "prn"                  : prn,
        "sampling_frequency"   : sampling_frequency,
        "quantization"         : quantization_bits,
        "cn0_target_dB"        : cn0_target_dB,
        "sigma_noise"          : sigma_noise,
        "signal_noisy"         : signal_noisy,
        "signal_quantized"     : signal_quantized,
        "scale_factor"         : scale_factor,
        "run"                  : run
    }

    # Perform exact correlations
    results[f"noisy_corr"], results[f"corr_lags"] = axg.PartialCorrelation(
        signal_noisy, replicas)
    results[f"quantized_corr"], _ = axg.PartialCorrelation(
        signal_quantized, replicas)
    
    # Perform axc correlations
    for name, axc_mult in custom_multipliers.items():
        results[f"axc_corr_{name}"], _ = axg.PartialCorrelation(
                signal_quantized, 
                replicas, 
                axc_mult=axc_mult, 
                axc_corr=0
            )
        
    print(f"Run done ({sampling_frequency:>12.1f} Hz, {quantization_bits:>2d} bits, {cn0_target_dB:>2d} dB, run {run:>4d})")

    return results

# ----------------------------------------------------------------------------------------------------------------------

def test_worker():

    sigma_noise = axg.getSigmaFromCN0(signal_power_dB=0, cn0_target_dB=60, signal_bw=axg.GPS_L1CA_CODE_FREQ)
    signal = axg.UpsampleCode(signal_prn, 10e6)
    replicas = axg.GenerateDelayedReplicas(signal, correlator_delays)
    all_results = worker(signal, replicas, 60, sigma_noise, 10e6, 8, axg.EAL_MULTIPLIERS_8BIT_SIGNED)
    
    # Save results
    with open('all_results.pkl', 'wb') as handle:
        pickle.dump(all_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Read results 
    with open('all_results.pkl', 'rb') as handle:
        all_results = pickle.load(handle)

    plt.figure()
    plt.plot(all_results['corr_lags'], all_results['noisy_corr'])
    plt.plot(all_results['corr_lags'], all_results['quantized_corr'])
    plt.plot(all_results['corr_lags'], all_results['axc_corr_mul8s_1L12'])
    plt.xlim((-20, 20))
    plt.grid()
    plt.savefig('test.png')

    return 

# ======================================================================================================================

if __name__ == "__main__":

    # ==================================================================================================================
    # Testing worker function
    # test_worker()
    
    # ==================================================================================================================
    # Parallel simulation processing
    # Simulation parameters
    sf_list = np.array([2, 4]) * axg.GPS_L1CA_CODE_FREQ
    cn0_list = range(30, 60, 5)
    bits_list = [8, 16]
    nb_run = 10 # Number Monte Carlo run per subset of parameter

    # Signal parameters
    prn = 1
    signal_prn = axg.GenerateGPSGoldCode(prn=prn)
    correlator_delays = range(-200, 201, 1) # in deca chip, chip * 10 (easier to handle integer range in python)
    
    # Multiprocessing
    num_processes = 4  # CPU cores
    pool = multiprocessing.Pool(processes=num_processes)

    # Simulations
    args = []
    for sf in sf_list:
        signal = axg.UpsampleCode(signal_prn, sf)
        # Generate the replicas
        replicas = axg.GenerateDelayedReplicas(signal, correlator_delays)

        for bits in bits_list:
            # Select approximate multipliers
            if bits == 8:
                axc_mults = axg.EAL_MULTIPLIERS_8BIT_SIGNED
            elif bits == 12:
                axc_mults = axg.EAL_MULTIPLIERS_12BIT_SIGNED
            elif bits == 16:
                axc_mults = axg.EAL_MULTIPLIERS_16BIT_SIGNED
            else:
                raise ValueError("Invalid quantization provided.")
            
            for cn0 in cn0_list:
                for run in range(nb_run):
                    args.append((run, prn, signal, replicas, cn0, sf, bits, axc_mults))

    # Launch processings
    tic = time.time()
    all_results = pool.starmap(worker, args)
    pool.close()
    pool.join()
    toc = time.time()
    print(f"Elapsed time {toc - tic}")

    # Save results
    with open('all_results.pkl', 'wb') as handle:
        pickle.dump(all_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("Done")