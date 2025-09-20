#!/bin/bash

export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=12345

TARGET_SCENARIO="$1"
shift
EXTRA_ARGS=("$@")

uv run python -m $TARGET_SCENARIO \
  --ckpt_dir ./checkpoints/Llama-3.2-1B-Instruct/original \
  --max_seq_len 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --max_batch_size 1 \
  "${EXTRA_ARGS[@]}"