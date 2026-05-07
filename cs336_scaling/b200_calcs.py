import math
def config_vals(n_original, C, num_layers=12):
    hidden_size = round(math.sqrt(n_original/(12 * num_layers)) / 64) * 64
    actual_n = 12*num_layers*hidden_size**2
    intermediate_d = round((8/3*hidden_size/64))*64
    step_size = 65536 * 5
    D = round((C/(6*actual_n))/step_size)*step_size
    num_heads = hidden_size //64
    print("n_original", n_original, hidden_size, actual_n, intermediate_d, D, num_heads)
    return hidden_size, actual_n, intermediate_d, D, num_heads

print(config_vals(401105022, 1.04e20, 12))
hidden_size, actual_n, intermediate_d, D, num_heads = config_vals(401105022, 1.04e20, 12)
def get_kaplan_lr(N):
    return 0.003239 -0.0001395*math.log(N) 
print(get_kaplan_lr(actual_n))

flops = 604126559245621
max_runtime = int(1.04e20/flops*2)
print(max_runtime)