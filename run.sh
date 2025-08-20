#!/bin/bash

export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=12345

EXTRA_ARGS=""
if [[ "$1" == "nokv" ]]; then
  echo "Disabling KV cache"
  EXTRA_ARGS="--disable_kv_cache"
fi

python -m models.llama3.scripts.failover_scenario \
  --ckpt_dir ./checkpoints/Llama-3.2-1B-Instruct/original \
  --max_seq_len 512 \
  --temperature 0.6 \
  --top_p 0.9 \
  --max_batch_size 4 \
  $EXTRA_ARGS