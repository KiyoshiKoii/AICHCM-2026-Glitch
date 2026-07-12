import requests
import json
import time

url = "http://127.0.0.0:8001/internal/search/visual"
# Note: FastAPI is binding to 0.0.0.0, we can access it via localhost or 127.0.0.1
url = "http://127.0.0.1:8001/internal/search/visual"

payload = {
    "visual_prompt": "TV news",
    "top_k": 5
}

headers = {
    "Content-Type": "application/json"
}

print(f"Sending POST request to {url}...")
print(f"Payload: {json.dumps(payload, indent=2)}")

start_time = time.time()
try:
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    
    results = response.json()
    end_time = time.time()
    
    print(f"\nResponse received in {end_time - start_time:.2f} seconds.")
    print(f"Status Code: {response.status_code}")
    print("Results:")
    print(json.dumps(results, indent=2))
    
except requests.exceptions.RequestException as e:
    print(f"Error calling API: {e}")
