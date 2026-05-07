#uv run cs336_scaling/submit_run.py --config configs/my_config.json
import requests
import argparse
import json

API_BASE_URL ="http://hyperturing.stanford.edu:8000"
headers = {"X-API-Key": "06777793"}

def main():
    with open("data/all_results.json", "r") as f:
        cur_results = json.load(f)
    new_results = []
    for result in cur_results:
        if result["loss"] is None:
            poll = requests.get(f"{API_BASE_URL}/experiment/{result['experiment_id']}", headers=headers).json()
            if poll["status"]["status_type"] == "completed":
                result["loss"] = poll["status"]['val_losses'][-1]
                result["status"] = "completed"
            elif poll["status"]["status_type"] == "failed":
                result["status"] = "failed"
            print(poll)
            result["used_runtime_seconds"] = poll["status"]["used_runtime_seconds"]
            
        new_results.append(result)
                
    with open("data/all_results.json", "w") as f:
        json.dump(new_results, f)
if __name__ == "__main__":
    main()
