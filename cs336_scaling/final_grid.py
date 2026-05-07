#uv run cs336_scaling/submit_run.py --config configs/my_config.json
import requests
import argparse
import json
import math
import numpy as np
API_BASE_URL ="http://hyperturing.stanford.edu:8000"
headers = {"X-API-Key": "06777793"}

def config_vals(n_original, C, num_layers=12):
    hidden_size = round(math.sqrt(n_original/(12 * num_layers)) / 64) * 64
    actual_n = 12*num_layers*hidden_size**2
    intermediate_d = round((8/3*hidden_size/64))*64
    step_size = 65536 * 5
    D = round((C/(6*actual_n))/step_size)*step_size
    num_heads = hidden_size //64
    print("n_original", n_original, hidden_size, actual_n, intermediate_d, D, num_heads)
    return hidden_size, actual_n, intermediate_d, D, num_heads

def get_kaplan_lr(N):
    return 0.003239 -0.0001395*math.log(N) 


def main(N, C):
    print("in main")
    with open(f"data/isoflop_{C:.0e}.json", "r") as f:
        cur_results = json.load(f)
        
    # all_original_N = [N, 2*N, 3*N, 4*N, 5*N]
    # all_original_N = [N, 2*N, 3*N]
    all_original_N = [2*N]
    # all_original_N = [5*N, 8*N]
    num_layers = 12
    flops = 604126559245621
    max_runtime = int(C/flops*1.5)
    for n in all_original_N:
        hidden_size, actual_n, intermediate_d, D, num_heads = config_vals(n, C, num_layers)
        # kaplan_lr = get_kaplan_lr(n)
        kaplan_lr = get_kaplan_lr(actual_n)
        training_config = {
            "architecture_config": {
                "attention_bias": False,
                "head_dim": 64,
                "hidden_size": hidden_size,
                "intermediate_size": intermediate_d,
                "num_attention_heads":num_heads,
                "num_hidden_layers": num_layers,
                "num_key_value_heads": num_heads,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000,
                "tie_word_embeddings": False,
                "dtype": "bfloat16",
                "vocab_size": 32000
            },
            "optimizer_config": {
                "lr_scheduler": {
                    "peak_value": kaplan_lr,
                    "final_lr_frac": 0.1,
                    "warmup_frac": 0.05,
                    "init_value": 0.0
                },

                "weight_decay": 1e-2,
                "beta1": 0.9,
                "beta2": 0.95,
                "eps": 1e-8,
                "eps_root": 1e-8,
                "grad_clip_norm": 1.0
            },
            "train_batch_size": 128,
            "val_batch_size": 32,
            "n_evals": 5,
            "total_train_tokens": int(D),
            "max_runtime_seconds": max_runtime,
            "model_seed": 0
        }
    
        result = requests.post(f"{API_BASE_URL}/submit", headers=headers, json=training_config).json()
        print(result)
    
        new_experiment = {
                "experiment_id": result["experiment_id"], 
                "parameters": actual_n,
                "data":D ,
                "compute_flops": 6*actual_n*D, 
                "loss": None,
                "status": None,
                "config": training_config
            }
        cur_results.append(new_experiment)
    with open(f"data/isoflop_{C:.0e}.json", "w") as f:
        json.dump(cur_results, f)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", type=float, required=True)
    parser.add_argument("--N", type=float, required=True)
    args = parser.parse_args()
    main(args.N, args.C)
    
    
