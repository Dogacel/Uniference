
export DEVICE=mps
export DEBUG=0

PROMPT_100 := $(shell cat ./checkpoints/prompt_1000.txt | head -n 100 | head -c 105)
PROMPT_200 := $(shell cat ./checkpoints/prompt_1000.txt | head -n 200 | head -c 105)
PROMPT_500 := $(shell cat ./checkpoints/prompt_1000.txt | head -n 500 | head -c 105)
PROMPT_1000 := $(shell cat ./checkpoints/prompt_1000.txt)
PROMPT_5000 := $(shell cat ./checkpoints/prompt_5000.txt)

all:
	./run.sh scenarios.voltage_scenario \
		--device_count=2 \
		--prompt="${PROMPT_1000}" \
		--disable_kv_cache \
		--max_seq_len=8192 \
		--max_tokens=1 \
		--program="experimental"

	./run.sh scenarios.voltage_scenario \
		--device_count=2 \
		--prompt="${PROMPT_1000}" \
		--disable_kv_cache \
		--max_seq_len=8192 \
		--max_tokens=1 \
		--program="voltage"


debug: clear_prof
	DEVICE=mps DEBUG=1 ./run.sh scenarios.voltage_scenario \
		--device_count=2 \
		--prompt="${PROMPT_1000}" \
		--disable_kv_cache \
		--max_seq_len=8192 \
		--max_tokens=1 \
		--program="experimental"
	
	uv run simsuite/trace_merger.py out.json profile_out --normalize-logical-clock

clear_prof:
	rm -rf profile_out
	mkdir profile_out

merge_prof:
	uv run simsuite/trace_merger.py out.json profile_out --normalize-logical-clock