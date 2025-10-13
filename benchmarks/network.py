import json
import numpy as np
import time
import torch
import torch.distributed as dist
import fire
import os

from scipy.optimize import curve_fit
from simsuite.network import model_broken


def train_model(rtt: float, bandwidth: float, knee: float, bytes: list, means: list):
    bandwidth = bandwidth * 1024 * 1024  # Convert MB/s to B/s
    print("Initial parameters:")
    print(f"RTT: {rtt}, Bandwidth: {bandwidth}, Knee: {knee}")

    beta_init = 1.0 / bandwidth
    p0 = [rtt, beta_init, beta_init, knee]
    bounds = (
        [rtt * 0.5, beta_init * 0.5, beta_init * 0.5, knee * 0.5],
        [rtt * 2, beta_init * 2, beta_init * 2, knee * 2],
    )

    x = bytes
    y = means
    params, _ = curve_fit(model_broken, x, y, p0=p0, bounds=bounds, maxfev=20000)

    (alpha, _, beta2, _) = params

    print("=== Fitted Network Model Parameters ===")
    print(f"Latency (alpha): {alpha:.6f} s")
    print(f"Bandwidth (large) ~ {1 / beta2 / 1e6:.2f} MB/s")

    params = params.tolist()
    print(f"params: {json.dumps(params)}")


def run(num_latency_tests=100, num_bw_tests=10, mode="send_recv"):
    dist.init_process_group(os.getenv("DIST_BACKEND", "gloo"))
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # ---- Ping (latency) test ----
    latencies: list[float] = []
    tensor = torch.zeros(1)
    for _ in range(num_latency_tests):
        if rank == 0:
            start = time.time()
            dist.send(tensor, dst=1)
            dist.recv(tensor, src=1)
            end = time.time()
            latencies.append((end - start) * 1000)
        elif rank == 1:
            dist.recv(tensor, src=0)
            dist.send(tensor, dst=0)
        dist.barrier()

    if rank == 0:
        mean = sum(latencies) / len(latencies)
        std = (sum((x - mean) ** 2 for x in latencies) / len(latencies)) ** 0.5
        print(f"[rank0] RTT (ms): min={min(latencies):.3f}, max={max(latencies):.3f}, mean={mean:.3f}, std={std:.3f}")

    dist.barrier()

    # ---- Bandwidth test with different tensor sizes ----
    sizes_bytes = [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
        262144,
        524288,
        1048576,
        2097152,
        4194304,
        8388608,
        12589824,
        16777216,
        24576512,
        33554432,
        41943040,
        50331648,
        67108864,
        83886080,
        104857600,
        125829120,
        167772160,
        201326592,
    ]  # 1B, 2B, 4B, 8B, ..., 200MB
    bandwidth_means = []
    sizes_means = []
    for size_bytes in sizes_bytes:
        num_floats = max(1, size_bytes // 4)  # float32 is 4 bytes
        big = torch.ones(num_floats, dtype=torch.float32)
        to_gather = [torch.zeros(num_floats, dtype=torch.float32) for _ in range(world_size)]
        bw_results = []
        time_results = []
        for _ in range(num_bw_tests):
            if rank == 0:
                start = time.time()

                if mode == "send_recv":
                    dist.send(big, dst=1)
                    dist.recv(big, src=1)
                elif mode == "all_gather":
                    dist.all_gather(to_gather, big)

                end = time.time()
                elapsed = end - start
                mbps = (size_bytes * 2) / (1024 * 1024) / elapsed  # send + recv, MB/s
                bw_results.append(mbps)
                time_results.append(elapsed)
            else:
                if mode == "send_recv":
                    dist.recv(big, src=0)
                    dist.send(big, dst=0)
                elif mode == "all_gather":
                    dist.all_gather(to_gather, big)
            dist.barrier()
            print("Sleeping for 0.1s to avoid overloading the network...")
            time.sleep(0.1)

        if rank == 0:
            mean_bw = sum(bw_results) / len(bw_results)
            std_bw = (sum((x - mean_bw) ** 2 for x in bw_results) / len(bw_results)) ** 0.5
            mean_time = sum(time_results) / len(time_results)
            std_time = (sum((x - mean_time) ** 2 for x in time_results) / len(time_results)) ** 0.5

            bandwidth_means.append(mean_bw)
            sizes_means.append(mean_time)

            if size_bytes < 1024:
                size_str = f"{size_bytes}B"
            elif size_bytes < 1024 * 1024:
                size_str = f"{size_bytes // 1024}KB"
            else:
                size_str = f"{size_bytes // (1024 * 1024)}MB"
            print(
                f"[rank0] Size={size_str} | Bandwidth (MB/s): min={min(bw_results):.2f}, max={max(bw_results):.2f}, mean={mean_bw:.2f}, std={std_bw:.2f}"
            )
            print(
                f"[rank0] Size={size_str} | Transfer time (s): min={min(time_results):.4f}, max={max(time_results):.4f}, mean={mean_time:.4f}, std={std_time:.4f}"
            )
        
        print(f"Sleeping for 1s to avoid overloading the network...")
        time.sleep(1.0)

    train_model(
        rtt=np.array(latencies).mean(),
        bandwidth=np.array(bandwidth_means).mean(),
        knee=64 * 1024,
        bytes=sizes_bytes,
        means=sizes_means,
    )


if __name__ == "__main__":
    fire.Fire(run)
    dist.destroy_process_group()
