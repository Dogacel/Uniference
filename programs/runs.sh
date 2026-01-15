
uv run benchmarks/real_tp_pp_poission.py --device_count 1 --pp_size 1 --batch_size 1

WORLD_BACKEND=simulation WORLD_SIZE=1 PP_SIZE=1 uv run benchmarks/real_tp_pp_poission.py --device_count 2 --pp_size 2 --batch_size 1

export RANK=0
export WORLD_SIZE=2
export PP_SIZE=2
export MASTER_ADDR=172.20.212.10
export MASTER_PORT=25012
export WORLD_BACKEND=pytorch
export DEVICE=cuda