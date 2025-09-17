
export DEVICE=mps
export DEBUG=0

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


debug:
	DEBUG=1 ./run.sh scenarios.voltage_scenario \
		--device_count=1 \
		--prompt="${PROMPT_1000}" \
		--disable_kv_cache \
		--max_seq_len=8192 \
		--max_tokens=1 \
		--program="experimental"
