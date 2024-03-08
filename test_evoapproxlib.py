import evoapproxlib as eal
import pandas as pd 
   
kpis = ["MAE_PERCENT", "MAE", "WCE_PERCENT", "WCE", "WCRE_PERCENT", "EP_PERCENT", 
        "MRE_PERCENT", "MSE", "PDK45_PWR", "PDK45_AREA", "PDK45_DELAY"]

# ===========================================

# module_list = []
# for name, module in eal.multipliers['8x8_signed'].items():
#     mdict = {"NAME": name}
#     for k in kpis:
#         mdict[k] = getattr(module, k)
#     module_list.append(mdict)
#     #print(f"{name} | {module.MAE:>6} | {module.WCE:>6} | {module.MSE:>9} | {module.PDK45_PWR:>9}")
# df = pd.DataFrame(module_list) 
# print(df)

# ===========================================

help(eal.mul8s_1L2D.calc)

