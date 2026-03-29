import requests
import json
import time
import sys

BASE_URL = "http://127.0.0.1:8000/api/v1"

def test_logistics():
    print("\n--- Testing Logistics Optimization ---")
    with open("data/sample_dataset_v3.json", "r") as f:
        payload = json.load(f)
    
    try:
        response = requests.post(f"{BASE_URL}/logistics/optimize", json=payload)
        if response.status_code == 200:
            print("Status: SUCCESS")
            print(f"Run ID: {response.json().get('run_id')}")
            print(f"Status: {response.json().get('status')}")
        else:
            print(f"Status: FAILED ({response.status_code})")
            print(response.text)
    except Exception as e:
        print(f"Error: {e}")

def test_portfolio():
    print("\n--- Testing Portfolio Optimization ---")
    payload = {
        "market_data": [
            {"asset_id": "AAPL", "expected_return": 0.12},
            {"asset_id": "GOOGL", "expected_return": 0.15},
            {"asset_id": "MSFT", "expected_return": 0.10}
        ],
        "constraints": {
            "budget": 100000,
            "risk_tolerance_lambda": 1.0,
            "max_weight": 0.5
        }
    }
    try:
        response = requests.post(f"{BASE_URL}/portfolio/optimize", json=payload)
        if response.status_code == 200:
            print("Status: SUCCESS")
            print(f"Selected Assets: {response.json().get('selected_assets')}")
            print(f"Sharpe Ratio: {response.json().get('sharpe_ratio')}")
        else:
            print(f"Status: FAILED ({response.status_code})")
            print(response.text)
    except Exception as e:
        print(f"Error: {e}")

def test_supply_chain():
    print("\n--- Testing Supply Chain Optimization ---")
    payload = {
        "nodes": [
            {"id": "Warehouse_A", "type": "warehouse", "stock": 100},
            {"id": "Store_B", "type": "store", "demand": 20}
        ],
        "edges": [
            {"source": "Warehouse_A", "target": "Store_B", "distance": 50}
        ]
    }
    try:
        response = requests.post(f"{BASE_URL}/supply-chain/optimize", json=payload)
            print("Status: SUCCESS")
            summary = response.json().get("business_impact", {}).get("summary", "No summary found")
            print(f"Optimization Summary: {summary}")
        else:
            print(f"Status: FAILED ({response.status_code})")
            print(response.text)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Give the server a moment to start if run concurrently
    print("Waiting for server to be ready...")
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=30)
        print(f"Health Check: {response.json()}")
    except Exception as e:
        print(f"Server not ready: {e}")
        sys.exit(1)

    test_logistics()
    test_portfolio()
    test_supply_chain()
