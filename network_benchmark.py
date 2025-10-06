import time
import torch
import torch.distributed as dist


def run(num_latency_tests=10, num_bw_tests=3):
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # ---- Ping (latency) test ----
    latencies = []
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
        16777216,
        33554432,
        104857600,
    ]  # 1B, 2B, 4B, 8B, ..., 100MB
    for size_bytes in sizes_bytes:
        num_floats = max(1, size_bytes // 4)  # float32 is 4 bytes
        big = torch.ones(num_floats, dtype=torch.float32)
        bw_results = []
        time_results = []
        for _ in range(num_bw_tests):
            if rank == 0:
                start = time.time()
                dist.send(big, dst=1)
                dist.recv(big, src=1)
                end = time.time()
                elapsed = end - start
                mbps = (size_bytes * 2) / (1024 * 1024) / elapsed  # send + recv, MB/s
                bw_results.append(mbps)
                time_results.append(elapsed)
            elif rank == 1:
                dist.recv(big, src=0)
                dist.send(big, dst=0)
            dist.barrier()

        if rank == 0:
            mean_bw = sum(bw_results) / len(bw_results)
            std_bw = (sum((x - mean_bw) ** 2 for x in bw_results) / len(bw_results)) ** 0.5
            mean_time = sum(time_results) / len(time_results)
            std_time = (sum((x - mean_time) ** 2 for x in time_results) / len(time_results)) ** 0.5
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


if __name__ == "__main__":
    run()
    dist.destroy_process_group()
