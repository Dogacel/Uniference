# gloo_ping_bw.py
import time
import torch
import torch.distributed as dist

def run():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # ---- Ping test ----
    tensor = torch.zeros(1)
    if rank == 0:
        start = time.time()
        dist.send(tensor, dst=1)
        dist.recv(tensor, src=1)
        end = time.time()
        print(f"[rank0] RTT ~ {(end - start)*1000:.3f} ms")
    elif rank == 1:
        dist.recv(tensor, src=0)
        dist.send(tensor, dst=0)

    dist.barrier()

    # ---- Bandwidth test ----
    size_mb = 100
    big = torch.ones(1024 * 1024 * size_mb // 4, dtype=torch.float32)  # ~100 MB tensor

    if rank == 0:
        start = time.time()
        dist.send(big, dst=1)
        dist.recv(big, src=1)
        end = time.time()
        mbps = (size_mb * 2) / (end - start)  # send + recv
        print(f"[rank0] Bandwidth ~ {mbps:.2f} MB/s")
    elif rank == 1:
        dist.recv(big, src=0)
        dist.send(big, dst=0)

if __name__ == "__main__":
    run()
    dist.destroy_process_group()