import pandas as pd
import evoapproxlib as eal
import matplotlib.pyplot as plt
from adjustText import adjust_text

# =============================================================================

def getMultipliersEAL(name):

    kpis = ["MAE_PERCENT", "MAE", "WCE_PERCENT", "WCE", "WCRE_PERCENT", "EP_PERCENT", 
        "MRE_PERCENT", "MSE", "PDK45_PWR", "PDK45_AREA", "PDK45_DELAY"]

    module_list = []
    for name, module in eal.multipliers[name].items():
        mdict = {"NAME": name}
        for k in kpis:
            mdict[k] = getattr(module, k)
        module_list.append(mdict)
        #print(f"{name} | {module.MAE:>6} | {module.WCE:>6} | {module.MSE:>9} | {module.PDK45_PWR:>9}")

    df = pd.DataFrame(module_list) 

    return df

# =============================================================================

def plotKPI(df, kpi_x, kpi_y, log_x=False, log_y=False):

    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.scatter(df['MAE_PERCENT'], df['PDK45_PWR'])
    ax.set_axisbelow(True)

    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
    
    # Add module names
    text = [ax.annotate(txt, (df['MAE_PERCENT'][i], df['PDK45_PWR'][i])) for i, txt in enumerate(df['NAME'])]

    adjust_text(text)

    _ = plt.xticks(rotation=45)
    plt.grid()
    plt.xlabel(kpi_x)
    plt.ylabel(kpi_y)
    
    return fig, ax

# =============================================================================

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