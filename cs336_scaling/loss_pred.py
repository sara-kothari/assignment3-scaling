import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def power_law(ND, E, A,alpha, B, beta):
    N,D = ND
    return E + A/N**alpha + B/D**beta

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
        
    params = []
    data_sizes = []
    losses = []


    for data in my_dfs:
        for entry in data:
            if entry["loss"] is None:
                continue
            N = entry["parameters"]
            D = entry["data"]
            loss = entry["loss"]
            losses.append(loss)
            params.append(N)
            data_sizes.append(D)
            
            
            
    param_arr = np.array(params)
    loss_arr = np.array(losses)
    data_arr = np.array(data_sizes)
    
    
    #N
    p0 = [1.69, 406.4, 0.34, 410.7, 0.28] 
    bounds = (0, np.inf) 
    popt, pcov = curve_fit(power_law, (param_arr, data_arr), loss_arr, p0=p0, bounds=bounds, maxfev=10000)
    E, A,alpha, B, beta = popt
    print(f"Fit result: loss =  {E:.4f} + {A:.4f}/N^{alpha:.4f} + {B:.4f}/D^{beta:.4f}")
    N_opt = 398721024
    # D_opt = 43472322560 
    # D_opt = 43471994880 
    D_opt = 33095482588
    predicted_loss = power_law((N_opt, D_opt), E, A, alpha, B, beta)
    print(f"Predicted loss: {predicted_loss}")
   
    
    predicted = power_law((param_arr, data_arr), *popt)
    plt.figure()
    plt.scatter(loss_arr, predicted, alpha=0.7)
    plt.plot([loss_arr.min(), loss_arr.max()], [loss_arr.min(), loss_arr.max()], 'r--', label='Perfect fit')
    plt.xlabel("Actual Loss")
    plt.ylabel("Predicted Loss")
    plt.title("Chinchilla Loss Fit: Predicted vs Actual")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_fit.png")
    ss_res = np.sum((loss_arr - predicted)**2)
    ss_tot = np.sum((loss_arr - loss_arr.mean())**2)
    r2 = 1 - ss_res/ss_tot
    print(f"R^2:{r2}")

if __name__ == "__main__":
    main()
        


        
        
        