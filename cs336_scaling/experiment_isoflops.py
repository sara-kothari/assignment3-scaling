import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def power_law(log_C, log_k, a):
    return log_k + a*log_C

def main():
    my_dfs = []
    with open("data/isoflop_1e+17.json", "r") as f:
        data = json.load(f)
        my_dfs.append(data)
    with open("data/isoflop_1e+18.json", "r") as f:
        data = json.load(f)
        my_dfs.append(data)
    with open("data/isoflop_5e+16.json", "r") as f:
        data = json.load(f)
        my_dfs.append(data)
    with open("data/isoflop_7e+17.json", "r") as f:
        data = json.load(f)
        my_dfs.append(data)
        

    def get_nominal_budget(compute_flops):
        nominal_budgets = [5e16, 1e17, 7e17, 1e18]
        return min(nominal_budgets, key=lambda x: abs(x - compute_flops))

    my_dict = {}
    for data in my_dfs:
        for entry in data:
            if entry["loss"] is None:
                continue
            nominal = get_nominal_budget(entry["compute_flops"])
            if nominal not in my_dict:
                my_dict[nominal] = {"best_loss": float("inf"), "best_n": None}
            if my_dict[nominal]["best_loss"] > entry["loss"]:
                my_dict[nominal]["best_loss"] = entry["loss"]
                my_dict[nominal]["best_n"] = entry["parameters"]
                my_dict[nominal]["best_d"] = entry["data"]
            
    budgets = []
    params = []
    data_sizes = []
    for budget in my_dict:
        budgets.append(budget)
        params.append(my_dict[budget]["best_n"])
        data_sizes.append(my_dict[budget]["best_d"])
    param_arr = np.array(params)
    budget_arr = np.array(budgets)
    data_arr = np.array(data_sizes)
    
    extra_c = budgets.copy()
    extra_c.extend([1e19,5e19, 1e20 ])
    extra_c = np.array(extra_c)
    
    #N
    popt, pcov = curve_fit(power_law, np.log(budget_arr), np.log(param_arr))
    logk_opt, a_opt = popt
    k_n = np.exp(logk_opt)
    print(f"Fit result: N = {k_n:.2f} * C^{a_opt:.2f}")
    plt.scatter(np.log(budget_arr), np.log(param_arr), label='Data')
    # plt.plot(budget_arr,k_n*budget_arr**a_opt , 'r-', label='Fit')
    print(f"For N: 5e19 = {k_n*5e19**a_opt}, 1.04e20 = {k_n*1.04e20**a_opt}")
    plt.plot(np.log(extra_c),np.log(k_n*extra_c**a_opt) , 'r-', label='Fit')
    plt.legend()
    plt.xlabel('log(C)')
    plt.ylabel('log(N)')
    plt.title(" N = 0.10 * C^0.48")
    plt.show()
    
    #D
    popt, pcov = curve_fit(power_law, np.log(budget_arr), np.log(data_arr))
    logk_opt, b_opt = popt
    k_d = np.exp(logk_opt)
    print(f"Fit result: D = {k_d:.2f} * C^{b_opt:.2f}")
    print(f"For D: 5e19 = {k_d*5e19**b_opt}, 1.04e20 = {k_d*1.04e20**b_opt}")
    
    plt.scatter(np.log(budget_arr), np.log(data_arr), label='Data')
    # plt.plot(budget_arr,k_d*budget_arr**b_opt , 'r-', label='Fit')
    plt.plot(np.log(extra_c),np.log(k_d*extra_c**b_opt) , 'r-', label='Fit')
    plt.legend()
    plt.xlabel('log(C)')
    plt.ylabel('log(D)')
    plt.title("D = 1.59 * C^0.52")
    plt.show()
    
    

if __name__ == "__main__":
    main()
        


        
        
        