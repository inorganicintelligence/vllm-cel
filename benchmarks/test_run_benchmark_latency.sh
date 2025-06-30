#!/bin/bash
#


python3 benchmark_latency.py \
	--model mistralai/Mixtral-8x7B-v0.1 \
	--input-len 512 \
	--output-len 128 \
	--batch-size 8 \
	--output-json test_out.json \
	--disable-detokenize \
	--tensor-parallel-size 4 \
	--enable-expert-parallel
