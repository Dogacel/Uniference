
uv run benchmarks/real_tp_pp_poission.py --device_count 1 --pp_size 1 --batch_size 1

WORLD_BACKEND=simulation WORLD_SIZE=1 PP_SIZE=1 uv run benchmarks/real_tp_pp_poission.py --device_count 2 --pp_size 2 --batch_size 2