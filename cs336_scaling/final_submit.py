import requests
import json
API_BASE_URL ="http://hyperturing.stanford.edu:8000"
headers = {"X-API-Key": "06777793"}
training_config = {
            "architecture_config": {
                "attention_bias": False,
                "head_dim": 64,
                "hidden_size": 1664,
                "intermediate_size": 4416,
                "num_attention_heads":26,
                "num_hidden_layers": 12,
                "num_key_value_heads": 26,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000,
                "tie_word_embeddings": False,
                "dtype": "bfloat16",
                "vocab_size": 32000
            },
            "optimizer_config": {
                "lr_scheduler": {
                    "peak_value": 0.0004763,
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
            "n_evals": 10,
            "total_train_tokens": 43471994880,
            "max_runtime_seconds": 172800,
            "model_seed": 0
        }

final_submission = {
    "training_config": training_config,
    "predicted_final_loss": 3.3092,
}

# result = requests.post(f"{API_BASE_URL}/final_submission",  headers=headers, json=final_submission).json()
# print(result)
print(requests.get(f"{API_BASE_URL}/final_submission", headers=headers).json())