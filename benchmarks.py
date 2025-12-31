import time
import requests
from PIL import Image
import torch
import torch.nn.functional as F
import os
import random

# -----------------------------
# RANDOM IMAGE SELECTION
# -----------------------------
def random_image(dir):
    image_paths = []
    
    if not os.path.isdir(dir):
        raise ValueError(f"Directory does not exist: {dir}")

    for root, _, files in os.walk(dir):
        for f in files:
            lower = f.lower()
            if lower.endswith(".jpg") or lower.endswith(".jpeg") or lower.endswith(".png"):
                full_path = os.path.join(root, f)
                image_paths.append(full_path)

    if not image_paths:
        raise RuntimeError(f"No images found under: {dir}")

    return random.choice(image_paths)


# -----------------------------
# 1. MODEL LOAD TIME
# -----------------------------
def benchmark_model_load(load_model_fn, model_path):
    start = time.perf_counter()
    model = load_model_fn(model_path)
    elapsed = time.perf_counter() - start
    print(f"[MODEL LOAD] {elapsed:.4f} seconds")
    return model

# -----------------------------
# 2. PREPROCESS + INFERENCE TIME
# -----------------------------
def benchmark_inference(model, transform, device, image_path, runs=50):
    preprocess_times = []
    inference_times = []

    for _ in range(runs):
        # Preprocessing
        t0 = time.perf_counter()
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
        preprocess_times.append(time.perf_counter() - t0)

        # Inference
        t1 = time.perf_counter()
        with torch.no_grad():
            _ = model(input_tensor)
        inference_times.append(time.perf_counter() - t1)

    print(f"[PREPROCESS] avg={sum(preprocess_times)/runs:.4f}s")
    print(f"[INFERENCE] avg={sum(inference_times)/runs:.4f}s")

# -----------------------------
# 3. END-TO-END LATENCY (Flask)
# -----------------------------
def benchmark_http(endpoint, image_path, runs=20):
    latencies = []

    for _ in range(runs):
        with open(image_path, "rb") as f:
            t0 = time.perf_counter()
            _ = requests.post(endpoint, files={"file": f})
            latencies.append(time.perf_counter() - t0)

    print(f"[HTTP END-TO-END] avg={sum(latencies)/runs:.4f}s")

# -----------------------------
# 4. THROUGHPUT TEST
# -----------------------------
def benchmark_throughput(model, transform, device, image_path, runs=100):
    start = time.perf_counter()
    for _ in range(runs):
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = model(input_tensor)
    total = time.perf_counter() - start

    print(f"[THROUGHPUT] {runs/total:.2f} images/sec")

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    from Code.views import load_model, transform, device  

    ROOT = os.path.dirname(os.path.abspath(__file__))
    
    IMAGE_PATH = random_image(ROOT + "\Code\data")  
    MODEL_PATH = "Code/models/mobilenetv2_trashnet2.pth"
    ENDPOINT = "http://127.0.0.1:80/" 

    print("\n=== BENCHMARKING ===\n")

    # 1. Model load
    model = benchmark_model_load(load_model, MODEL_PATH)

    # 2. Preprocess + inference
    benchmark_inference(model, transform, device, IMAGE_PATH)

    # 3. End-to-end HTTP latency
    benchmark_http(ENDPOINT, IMAGE_PATH)

    # 4. Throughput
    benchmark_throughput(model, transform, device, IMAGE_PATH)