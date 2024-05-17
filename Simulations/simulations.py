import axcgnss as axg

import matplotlib.pyplot as plt

import pickle

# =================================================================================================
# Signal parameters
prn = 1
signal_prn = axg.GenerateGPSGoldCode(prn=prn)
correlator_delays = range(-200, 250, 1) # in deca chip, chip * 10

# =================================================================================================

def worker(signal, replicas, cn0_target_dB, sigma_noise, sampling_frequency, quantization_bits, custom_multipliers):

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
        "scale_factor"         : scale_factor
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

    return results

if __name__ == "__main__":

    # Parameters
    # sf_list = [2, 4, 8, 16, 20] * axg.GPS_L1CA_CODE_FREQ
    # cn0_list = range(30, 60, 5)
    # bits_list = [8, 16]
    # nb_run = 100 # Number Monte Carlo run per subset of parameter
    # signal_bw = axg.GPS_L1CA_CODE_FREQ # Bandwidth of the signal for CN0 to SNR computation

    # Testing worker function
    # sigma_noise = axg.getSigmaFromCN0(signal_power_dB=0, cn0_target_dB=60, signal_bw=axg.GPS_L1CA_CODE_FREQ)
    # signal = axg.UpsampleCode(signal_prn, 10e6)
    # replicas = axg.GenerateDelayedReplicas(signal, correlator_delays)
    # all_results = worker(signal, replicas, 60, sigma_noise, 10e6, 8, axg.EAL_MULTIPLIERS_8BIT_SIGNED)

    # with open('results.pkl', 'wb') as handle:
    #     pickle.dump(all_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # # "mul8s_1L12" : eal.mul8s_1L12.calc
    # print("Done")
    

    # plotCorrelation({'AxC': (all_results['signal_axc_corr_mul8s_1L12'], all_results['signal_axc_lags_mul8s_1L12'])}, 10e6, axg.GPS_L1CA_CODE_FREQ, f"PRN G{prn:02d}, GPS L1 C/A")
    # plt.savefig('test.png')

    # Parallel simulation processing
    # sf_list = [10] * GPS_L1CA_CODE_FREQ
    # cn0_list = [60]
    # bits_list = [8, 16]
    # nb_run = 1 # Number Monte Carlo run per subset of parameter
    # signal_bw = GPS_L1CA_CODE_FREQ # Bandwidth of the signal for CN0 to SNR computation
    # # Select custom multipliers
    # custom_multipliers = {}
    # for name, module in eal.multipliers['8x8_signed'].items():
    #     custom_multipliers[name] = getattr(module, 'calc')
    # for name, module in eal.multipliers['16x16_signed'].items():
    #     custom_multipliers[name] = getattr(module, 'calc')
    
    # # Multiprocessing
    # num_processes = 4  # CPU cores
    # pool = multiprocessing.Pool(processes=num_processes)

    # # Simulations
    # args = []
    # for sf in sf_list:
    #     for bits in bits_list:
    #         _custom_mult = {k: v for k, v in custom_multipliers.items() if k.startswith(f'{bits}')}
    #         for cn0 in cn0_list:
    #             for _ in nb_run:
    #                 sigma_noise = getSigmaFromCN0(signal_power_dB=0, cn0_target_dB=cn0, signal_bw=signal_bw)
    #                 args.append(sigma_noise, sf, bits, _custom_mult)

    # results = pool.starmap(worker, args)

    # pool.close()
    # pool.join()
    
    print("Done")