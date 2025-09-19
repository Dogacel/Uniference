#!/bin/bash

PROMPT_100=$(cat ./checkpoints/prompt_1000.txt | head -n 100 | head -c 100)
PROMPT_200=$(cat ./checkpoints/prompt_1000.txt | head -n 200 | head -c 200)
PROMPT_500=$(cat ./checkpoints/prompt_1000.txt | head -n 500 | head -c 500)
PROMPT_1000=$(cat ./checkpoints/prompt_1000.txt | head -n 1000 | head -c 1000)

trap "echo 'Interrupted! Killing all child processes...'; kill 0; exit 1" SIGINT

for i in $(seq 1 10); do
    DEVICE=mps DEBUG=1 ./run.sh scenarios.voltage_scenario \
        --device_count=2 \
        --prompt="${PROMPT_1000}" \
        --disable_kv_cache \
        --max_seq_len=8192 \
        --max_tokens=10 \
        --program="experimental";

    DEVICE=mps DEBUG=1 ./run.sh scenarios.voltage_scenario \
        --device_count=2 \
        --prompt="${PROMPT_1000}" \
        --disable_kv_cache \
        --max_seq_len=8192 \
        --max_tokens=10 \
        --program="voltage";
done