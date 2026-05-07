#uv run cs336_scaling/get_budget.py
import requests
import argparse
import json

API_BASE_URL ="http://hyperturing.stanford.edu:8000"
headers = {"X-API-Key": "06777793"}


if __name__ == "__main__":
   print(requests.get(f"{API_BASE_URL}/budget", headers=headers).json())
