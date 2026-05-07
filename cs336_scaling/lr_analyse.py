import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


#LR = k + a*log(N)
def power_law(N, k, a):
    return k + a*np.log(N)

def main():
    with open("data/lr_results.json", "r") as f:
        data = json.load(f)
        
    all_results_dict = {}
    for entry in data:
        if entry["parameters"] not in all_results_dict:
            all_results_dict[entry["parameters"]] = []
        all_results_dict[entry["parameters"]].append((entry["lr"], entry["loss"]))
        
    print(all_results_dict)
    my_dict = {}
    for entry in data:
        if entry["parameters"] not in my_dict:
            my_dict[entry["parameters"]] = {"best_loss": float("inf"), "best_lr": None}
        if  my_dict[entry["parameters"]]["best_loss"] > entry["loss"]:
            my_dict[entry["parameters"]]["best_loss"] = entry["loss"]
            my_dict[entry["parameters"]]["best_lr"] = entry["lr"]
    

    params = []
    lrs = []
    
    
    for param in my_dict:
        params.append(param)
        lrs.append(my_dict[param]["best_lr"])
    print(lrs)
    param_arr = np.array(params)
    lr_arr = np.array(lrs)
    
    # extra_c = budgets.copy()
    # extra_c.extend([1e22, 1e23, 1e24])
    # extra_c = np.array(extra_c)
    
    #N vs lr
    popt, pcov = curve_fit(power_law, param_arr, lr_arr)
    k_opt, a_opt = popt
    print(f"Fit result: LR= {k_opt:.2f}  + {a_opt:.2f} * log(N)")
    plt.scatter(param_arr, lr_arr, label='Data')
    # plt.plot(budget_arr,k_n*budget_arr**a_opt , 'r-', label='Fit')
    plt.plot(param_arr,power_law(param_arr, k_opt, a_opt) , 'r-', label='Fit')
    plt.legend()
    plt.ylabel('LR')
    plt.xlabel('N')
    # plt.title("N = 1.16 * C^0.47")
    plt.show()
    

    

if __name__ == "__main__":
    main()
        


        
        
        