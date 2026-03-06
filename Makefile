
export DEVICE=mps
export DEBUG=0

export RANK=0
export WORLD_SIZE=1
export MASTER_PORT=29500
export MASTER_ADDR=localhost

RUN_ID=$(shell date +%s)

sanity:
	DEVICE=cpu uv run benchmarks/real_tp.py --text-lengths 128 --max-tokens 8 --repeats 1

sanity_cuda:
	DEVICE=cuda uv run benchmarks/real_tp.py --text-lengths 128 --max-tokens 8 --repeats 1

sanity_mps:
	DEVICE=mps uv run benchmarks/real_tp.py --text-lengths 128 --max-tokens 8 --repeats 1

clear_prof:
	rm -rf profile_out
	mkdir profile_out

merge_prof:
	uv run simsuite/trace_merger.py out.json profile_out --normalize-logical-clock
