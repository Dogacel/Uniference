#!/bin/bash

PROMPT_100=$(head -c 100 ./checkpoints/prompt_1000.txt)
PROMPT_200=$(head -c 200 ./checkpoints/prompt_1000.txt)
PROMPT_500=$(head -c 500 ./checkpoints/prompt_1000.txt)
PROMPT_1000=$(cat ./checkpoints/prompt_1000.txt)
PROMPT_5000=$(cat ./checkpoints/prompt_5000.txt)

if [ -z "$MASTER_ADDR" ] ; then
  export MASTER_ADDR=localhost
fi

if [ -z "$MASTER_PORT" ]; then
  export MASTER_PORT=25001
fi

uv run python -m scenarios.tensor_parallel_scenario \
  --ckpt_dir ./checkpoints/Llama-3.2-1B-Instruct/original \
  --temperature 0.0 \
  --world_size $WORLD_SIZE \
  --top_p 1.0 \
  --max_batch_size 1 \
  --device_count=1 \
  --prompt="${PROMPT_200}" \
  --max_seq_len=4096 \
  --max_tokens=25 \
  --backend="pytorch"