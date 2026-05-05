import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def power_law(log_C, log_k, a):
    return log_k + a*log_C

def main():
    with open("data/isoflops_curves.json", "r") as f:
        data = json.load(f)
        
    
    my_dict = {}
    for entry in data:
        if entry["compute_budget"] not in my_dict:
            my_dict[entry["compute_budget"]] = {"best_loss": float("inf"), "best_n": None}
        if my_dict[entry["compute_budget"]]["best_loss"] > entry["final_loss"]:
            my_dict[entry["compute_budget"]]["best_loss"] = entry["final_loss"]
            my_dict[entry["compute_budget"]]["best_n"] = entry["parameters"]
    
    budgets = []
    params = []
    data_sizes = []
    for budget in my_dict:
        budgets.append(budget)
        params.append(my_dict[budget]["best_n"])
        data_sizes.append(budget/my_dict[budget]["best_n"]/6)
    param_arr = np.array(params)
    budget_arr = np.array(budgets)
    data_arr = np.array(data_sizes)
    
    extra_c = budgets.copy()
    extra_c.extend([1e22, 1e23, 1e24])
    extra_c = np.array(extra_c)
    
    #N
    popt, pcov = curve_fit(power_law, np.log(budget_arr), np.log(param_arr))
    logk_opt, a_opt = popt
    k_n = np.exp(logk_opt)
    print(f"Fit result: N = {k_n:.2f} * C^{a_opt:.2f}")
    plt.scatter(np.log(budget_arr), np.log(param_arr), label='Data')
    # plt.plot(budget_arr,k_n*budget_arr**a_opt , 'r-', label='Fit')
    print(f"For N: 1e23 = {k_n*1e23**a_opt}, 1e24 = {k_n*1e24**a_opt}")
    plt.plot(np.log(extra_c),np.log(k_n*extra_c**a_opt) , 'r-', label='Fit')
    plt.legend()
    plt.xlabel('log(C)')
    plt.ylabel('log(N)')
    plt.title("N = 1.16 * C^0.47")
    plt.show()
    
    #D
    popt, pcov = curve_fit(power_law, np.log(budget_arr), np.log(data_arr))
    logk_opt, b_opt = popt
    k_d = np.exp(logk_opt)
    print(f"Fit result: D = {k_d:.2f} * C^{b_opt:.2f}")
    print(f"For D: 1e23 = {k_d*1e23**b_opt}, 1e24 = {k_d*1e24**b_opt}")
    
    plt.scatter(np.log(budget_arr), np.log(data_arr), label='Data')
    # plt.plot(budget_arr,k_d*budget_arr**b_opt , 'r-', label='Fit')
    plt.plot(np.log(extra_c),np.log(k_d*extra_c**b_opt) , 'r-', label='Fit')
    plt.legend()
    plt.xlabel('log(C)')
    plt.ylabel('log(D)')
    plt.title("D = 0.14 * C^0.53")
    plt.show()
    
    

if __name__ == "__main__":
    main()
        


        
        
        