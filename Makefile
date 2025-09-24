
export DEVICE=mps
export DEBUG=0

PROMPT_100 := $(shell cat ./checkpoints/prompt_1000.txt | head -c 100)
PROMPT_200 := $(shell cat ./checkpoints/prompt_1000.txt | head -c 200)
PROMPT_500 := $(shell cat ./checkpoints/prompt_1000.txt | head -c 500)
PROMPT_1000 := $(shell cat ./checkpoints/prompt_1000.txt)
PROMPT_5000 := $(shell cat ./checkpoints/prompt_5000.txt)

RUN_ID=$(shell date +%s)

pdebug: clear_prof
	DEVICE=mps DEBUG=1 ./run.sh scenarios.yield_perf_scenario \
		--device_count=1 \
		--prompt="${PROMPT_1000}" \
		--max_seq_len=4096 \
		--max_tokens=25 \
		--yield_probability=1.0 \
# 		--performance_mode \
# 		--debug_run \

	uv run simsuite/trace_merger.py results/${RUN_ID}_yield_perf.json profile_out --normalize-logical-clock

debug: clear_prof
    # Warmup 
	DEVICE=mps DEBUG=1 ./run.sh scenarios.voltage_scenario \
		--device_count=1 \
		--prompt="Hello!" \
		--disable_kv_cache \
		--max_seq_len=512 \
		--max_tokens=10 \
		--program="voltage"
	rm -rf profile_out

	# Actual run
	DEVICE=mps DEBUG=1 ./run.sh scenarios.voltage_scenario \
		--device_count=2 \
		--prompt="${PROMPT_1000}" \
		--disable_kv_cache \
		--max_seq_len=8192 \
		--max_tokens=1 \
		--program="voltage"

	uv run simsuite/trace_merger.py results/${RUN_ID}_voltage.json profile_out --normalize-logical-clock

	rm -rf profile_out

	DEVICE=mps DEBUG=1 ./run.sh scenarios.voltage_scenario \
		--device_count=2 \
		--prompt="${PROMPT_1000}" \
		--disable_kv_cache \
		--max_seq_len=8192 \
		--max_tokens=1 \
		--program="experimental"

	uv run simsuite/trace_merger.py results/${RUN_ID}_experimental.json profile_out --normalize-logical-clock


clear_prof:
	rm -rf profile_out
	mkdir profile_out

merge_prof:
	uv run simsuite/trace_merger.py out.json profile_out --normalize-logical-clock