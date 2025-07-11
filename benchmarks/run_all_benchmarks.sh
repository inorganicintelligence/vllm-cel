#!/bin/bash

# Define the parameter space to search
INPUT_LENS=(512 1024)
OUTPUT_LENS=(512 2048)
BATCH_SIZES=(8)
TP_SIZES=(2) # Assuming always using 2 GPUs for Mixtral
EP_MODES=(true) # true for --enable-expert-parallel, false for not

# CSV file to store results
RESULTS_CSV="logs/testbenchmark_results.csv"

# Write the header to the CSV file
echo "input_len,output_len,batch_size,tp_size,ep_enabled,system_throughput,request_throughput,avg_prefill_time,avg_decode_time,mean_ttft,mean_tpot,itl_mean,itl_median,itl_p99" > "$RESULTS_CSV"

# Loop through all parameter combinations
for input_len in "${INPUT_LENS[@]}"; do
  for output_len in "${OUTPUT_LENS[@]}"; do
    for batch_size in "${BATCH_SIZES[@]}"; do
      for tp_size in "${TP_SIZES[@]}"; do
        for ep_enabled in "${EP_MODES[@]}"; do
          
          echo "Running benchmark with: input_len=$input_len, output_len=$output_len, batch_size=$batch_size, tp_size=$tp_size, ep_enabled=$ep_enabled"

          # Build the command
          CMD="python3 benchmark_performance.py \
              --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
              --input-len $input_len \
              --output-len $output_len \
              --batch-size $batch_size \
              --tensor-parallel-size $tp_size \
              --num-prompts 32 \
              --num-warmup-runs 1 \
              --output-format json"

          if [ "$ep_enabled" = true ]; then
            CMD="$CMD --enable-expert-parallel"
          fi

          # Run the command and capture the JSON output
          # Using stdbuf to disable output buffering for immediate feedback
          raw_output=$(stdbuf -o0 $CMD 2>&1)
          
          # Debug: Show the raw output
          echo "DEBUG: Raw output from benchmark command:"
          echo "$raw_output"
          echo "DEBUG: End of raw output"
          
          # The python script might output some warnings or info messages, so we
          # grep for the JSON output line. It is assumed that the JSON output is
          # the last line that contains a '{' character.
          json_output=$(echo "$raw_output" | grep '^{' | tail -n 1)
          
          echo "DEBUG: Extracted JSON output: '$json_output'"
          
          # Check if the command was successful and produced valid JSON
          if [ -z "$json_output" ]; then
              echo "Error: No JSON output found in the command output."
              echo "Continuing to check if we should write a row with error values..."
              # Write a row with error values to indicate the benchmark failed
              echo "$input_len,$output_len,$batch_size,$tp_size,$ep_enabled,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR" >> "$RESULTS_CSV"
              echo "Error row written to $RESULTS_CSV"
              continue
          fi
          
          # Check if jq is available
          if ! command -v jq &> /dev/null; then
              echo "Warning: jq is not installed. Using Python for JSON parsing."
              # Use Python to parse JSON
              read -r system_throughput request_throughput avg_prefill_time avg_decode_time mean_ttft mean_tpot itl_mean itl_median itl_p99 <<< $(python3 -c "
import json
import sys
data = json.loads('$json_output')
print(
    data['throughput']['system_throughput'],
    data['throughput']['request_throughput'],
    data['latency']['avg_prefill_time'],
    data['latency']['avg_decode_time'],
    data['latency']['ttft']['mean'],
    data['latency']['tpot']['mean'],
    data['latency']['itl']['mean'],
    data['latency']['itl']['median'],
    data['latency']['itl']['p99']
)
" 2>/dev/null)
              
              if [ -z "$system_throughput" ]; then
                  echo "Error: Failed to parse JSON output"
                  echo "$input_len,$output_len,$batch_size,$tp_size,$ep_enabled,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR" >> "$RESULTS_CSV"
                  continue
              fi
          else
              # Parse the JSON output using jq
              if ! echo "$json_output" | jq . > /dev/null 2>&1; then
                  echo "Error: Command produced invalid JSON for the above configuration."
                  echo "JSON output was: $json_output"
                  # Write a row with error values to indicate the benchmark failed
                  echo "$input_len,$output_len,$batch_size,$tp_size,$ep_enabled,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR" >> "$RESULTS_CSV"
                  echo "Error row written to $RESULTS_CSV"
                  continue
              fi
              
              system_throughput=$(echo "$json_output" | jq .throughput.system_throughput)
              request_throughput=$(echo "$json_output" | jq .throughput.request_throughput)
              avg_prefill_time=$(echo "$json_output" | jq .latency.avg_prefill_time)
              avg_decode_time=$(echo "$json_output" | jq .latency.avg_decode_time)
              mean_ttft=$(echo "$json_output" | jq .latency.ttft.mean)
              mean_tpot=$(echo "$json_output" | jq .latency.tpot.mean)
              itl_mean=$(echo "$json_output" | jq .latency.itl.mean)
              itl_median=$(echo "$json_output" | jq .latency.itl.median)
              itl_p99=$(echo "$json_output" | jq .latency.itl.p99)
          fi

          # Write the results to the CSV file
          echo "$input_len,$output_len,$batch_size,$tp_size,$ep_enabled,$system_throughput,$request_throughput,$avg_prefill_time,$avg_decode_time,$mean_ttft,$mean_tpot,$itl_mean,$itl_median,$itl_p99" >> "$RESULTS_CSV"

          echo "Results appended to $RESULTS_CSV"
          echo "---------------------------------"
        done
      done
    done
  done
done

echo "All benchmarks finished. Results are in $RESULTS_CSV" 