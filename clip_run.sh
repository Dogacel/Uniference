#!/bin/bash

export RANK=0
export WORLD_SIZE=1

if [ -z "$MASTER_ADDR" ] ; then
  export MASTER_ADDR=localhost
fi

if [ -z "$MASTER_PORT" ]; then
  export MASTER_PORT=25001
fi

uv run python -m scenarios.clip_perf_scenario \
  --device_count=1 \
  --backend="simulation"