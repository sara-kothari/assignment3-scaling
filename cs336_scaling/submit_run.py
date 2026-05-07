#uv run cs336_scaling/submit_run.py --config configs/my_config.json
import requests
import argparse
import json

API_BASE_URL ="http://hyperturing.stanford.edu:8000"
headers = {"X-API-Key": "06777793"}

def main(training_config):
    result = requests.post(f"{API_BASE_URL}/submit", headers=headers, json=training_config).json()
    N = 12*training_config["architecture_config"]["num_hidden_layers"]*training_config["architecture_config"]["hidden_size"]**2
    D = training_config["total_train_tokens"]
    print(result)
    with open("data/all_results.json", "r") as f:
        cur_results = json.load(f)
    new_experiment = {
            "experiment_id": result["experiment_id"], 
            "parameters": N,
            "data":D ,
            "compute_flops": 6*N*D, 
            "loss": None,
            "status": None,
            "config": training_config
        }
    cur_results.append(new_experiment)
    with open("data/all_results.json", "w") as f:
        json.dump(cur_results, f)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)
    main(config)
