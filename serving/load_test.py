import time
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor

API_URL = "http://localhost:8000/predict"

# ตัวอย่างข้อมูลจำลอง
SAMPLE_PAYLOAD = {
    "Time": 0.0,
    "V1": -1.359, "V2": -0.072, "V3": 2.536, "V4": 1.378, "V5": -0.338,
    "V6": 0.462, "V7": 0.239, "V8": 0.098, "V9": 0.363, "V10": 0.090,
    "V11": -0.551, "V12": -0.617, "V13": -0.991, "V14": -0.311, "V15": 1.468,
    "V16": -0.470, "V17": 0.207, "V18": 0.025, "V19": 0.403, "V20": 0.251,
    "V21": -0.018, "V22": 0.277, "V23": -0.110, "V24": 0.066, "V25": 0.128,
    "V26": -0.189, "V27": 0.133, "V28": -0.021, "Amount": 149.62
}

def send_request():
    start = time.time()
    try:
        response = requests.post(API_URL, json=SAMPLE_PAYLOAD, timeout=2)
        status = response.status_code
    except:
        status = 500
    latency = (time.time() - start) * 1000 # ms
    return latency, status

def run_test(concurrent_requests, total_requests=1000):
    print(f"\n--- Running Load Test with {concurrent_requests} concurrent requests ---")
    latencies = []
    success = 0
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        results = executor.map(lambda _: send_request(), range(total_requests))
        
    for latency, status in results:
        latencies.append(latency)
        if status == 200:
            success += 1

    total_time = time.time() - start_time
    rps = total_requests / total_time
    error_rate = ((total_requests - success) / total_requests) * 100
    
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)

    print(f"Total Requests: {total_requests}")
    print(f"RPS (Requests/sec): {rps:.2f}")
    print(f"Error Rate: {error_rate:.2f}%")
    print(f"Latency P50: {p50:.2f} ms")
    print(f"Latency P95: {p95:.2f} ms  <-- (SLO Target: < 100 ms)")
    print(f"Latency P99: {p99:.2f} ms")

if __name__ == "__main__":
    # ยิงทดสอบ 3 รูปแบบตามที่กำหนดในโครงงาน
    run_test(concurrent_requests=1, total_requests=100)
    time.sleep(2)
    run_test(concurrent_requests=10, total_requests=500)
    time.sleep(2)
    run_test(concurrent_requests=100, total_requests=1000)