
export DEVICE=mps
export DEBUG=0

PROMPT_100 := $(shell cat ./checkpoints/prompt_1000.txt | head -c 100)
PROMPT_200 := $(shell cat ./checkpoints/prompt_1000.txt | head -c 200)
PROMPT_500 := $(shell cat ./checkpoints/prompt_1000.txt | head -c 500)
PROMPT_1000 := $(shell cat ./checkpoints/prompt_1000.txt)
PROMPT_5000 := $(shell cat ./checkpoints/prompt_5000.txt)

RUN_ID=$(shell date +%s)

profile:
	DEVICE=cpu ./run.sh scenarios.concurrent_scenario \
		--device_count=2 \
		--prompt="${PROMPT_500}" \
		--max_seq_len=4096 \
		--max_tokens=1 \
		--yield_probability=1.0 \
		--debug_run=1

	uv run simsuite/trace_merger.py results/${RUN_ID}_yield_perf.json profile_out --normalize-logical-clock

sanity:
	DEVICE=cpu ./run.sh scenarios.yield_perf_scenario \
		--device_count=1 \
		--prompt="${PROMPT_100}" \
		--max_seq_len=512 \
		--max_tokens=10 \
		--yield_probability=1.0

sanity_cuda:
	DEVICE=cuda ./run.sh scenarios.yield_perf_scenario \
		--device_count=1 \
		--prompt="${PROMPT_100}" \
		--max_seq_len=512 \
		--max_tokens=10 \
		--yield_probability=1.0

sanity_mps:
	DEVICE=mps ./run.sh scenarios.yield_perf_scenario \
		--device_count=1 \
		--prompt="${PROMPT_100}" \
		--max_seq_len=512 \
		--max_tokens=10 \
		--yield_probability=1.0

sanity_voltage:
	DEVICE=cpu ./run.sh scenarios.voltage_scenario \
		--device_count=1 \
		--prompt="${PROMPT_100}" \
		--max_seq_len=512 \
		--max_tokens=1 \
		--program=voltage

sanity_voltage_2: clear_prof
	DEVICE=cpu ./run.sh scenarios.voltage_scenario \
		--device_count=2 \
		--prompt="${PROMPT_100}" \
		--max_seq_len=512 \
		--max_tokens=1 \
		--program=voltage


sanity_voltage_4:
	DEVICE=mps ./run.sh scenarios.voltage_scenario \
		--device_count=4 \
		--prompt="${PROMPT_1000}" \
		--max_seq_len=4096 \
		--max_tokens=1 \
		--program=voltage

clear_prof:
	rm -rf profile_out
	mkdir profile_out

merge_prof:
	uv run simsuite/trace_merger.py out.json profile_out --normalize-logical-clock
