#!/bin/bash

python3 benchmark_performance.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --input-len 2048 \
    --output-len 2048 \
    --batch-size 8 \
    --num-prompts 32 \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
    --num-warmup-runs 2 \
