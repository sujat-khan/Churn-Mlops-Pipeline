"""
Simulate realistic production inference traffic by sending batches of records
from test.csv to the running FastAPI endpoint.
"""
import time
import requests
import pandas as pd

API_URL = "http://localhost:8000/predict"
TEST_DATA_PATH = "data/raw/test.csv"


def simulate_production_traffic(num_records: int = 40):
    print(f"Loading {num_records} records from {TEST_DATA_PATH}...")
    df = pd.read_csv(TEST_DATA_PATH)

    # Exclude target column
    feature_cols = [c for c in df.columns if c != "Attrition"]
    sample_df = df[feature_cols].head(num_records)

    records = sample_df.to_dict(orient="records")
    print(f"Sending {len(records)} prediction requests to {API_URL}...")

    success_count = 0
    for i, record in enumerate(records, 1):
        try:
            res = requests.post(API_URL, json=record, timeout=5)
            if res.status_code == 200:
                success_count += 1
            if i % 10 == 0 or i == len(records):
                print(f"  Processed {i}/{len(records)} requests...")
            time.sleep(0.05)  # small pause to simulate incoming traffic
        except Exception as e:
            print(f"  Error sending request {i}: {e}")
            break

    print(f"\nCompleted: {success_count}/{num_records} predictions recorded.")
    print("Check data/monitoring/production_inferences.csv to see the logged records!")


if __name__ == "__main__":
    simulate_production_traffic(40)
