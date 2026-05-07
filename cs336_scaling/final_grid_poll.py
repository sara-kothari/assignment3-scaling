#uv run cs336_scaling/submit_run.py --config configs/my_config.json
import requests
import argparse
import json

API_BASE_URL ="http://hyperturing.stanford.edu:8000"
headers = {"X-API-Key": "06777793"}

def main():
    with open("data/isoflop_7e+17.json", "r") as f:
        cur_results = json.load(f)
    new_results = []
    for result in cur_results:
        if result["loss"] is None:
            poll = requests.get(f"{API_BASE_URL}/experiment/{result['experiment_id']}", headers=headers).json()
            print(poll)
            if poll["status"]["status_type"] == "completed":
                result["loss"] = poll["status"]['val_losses'][-1]
                result["status"] = "completed"
                result["used_runtime_seconds"] = poll["status"]["used_runtime_seconds"]
            # elif poll["status"]["status_type"] == "failed":
            #     continue
            # if "used_runtime_seconds" not in poll["status"]:
            #     continue
            
        
        new_results.append(result)
    
    # for id in [4405,4406,4407,4408, 4409, 4410 ]:
    #     poll = requests.get(f"{API_BASE_URL}/experiment/{id}", headers=headers).json()
    #     print(poll)
    #     N = poll["training_config"]["architecture_config"]['num_hidden_layers']*12*poll["training_config"]["architecture_config"]['hidden_size']**2
    #     D = poll["training_config"]["total_train_tokens"]
    #     result = {
    #         "experiment_id": id, 
    #         "lr": poll["training_config"]["optimizer_config"]["lr_scheduler"]["peak_value"],
    #         "parameters": N,
    #         "data":D ,
    #         "compute_flops": 6*N*D, 
    #         "loss": None,
    #         "status": None,
    #         "config": poll["training_config"]
    #     }
    #     if poll["status"]["status_type"] == "completed":
    #         result["loss"] = poll["status"]['val_losses'][-1]
    #         result["status"] = "completed"
    #     result["used_runtime_seconds"] = poll["status"]["used_runtime_seconds"]
    #     new_results.append(result)
    
            
    
    
    
                
    with open("data/isoflop_7e+17.json", "w") as f:
        json.dump(new_results, f, indent=2)
if __name__ == "__main__":
    main()
