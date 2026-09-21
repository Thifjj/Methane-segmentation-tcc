#!/bin/bash
set -e

'/home/root/Benchmark_ZCU104/benchmark_zcu104' --profile max-e2e --model '/home/root/models/compiled_zcu104/baseline/methane_baseline.xmodel' --dataset '/home/root/STARCOP_test' --csv '/home/root/STARCOP_test/test.csv' --out '/home/root/Benchmark_ZCU104/results_all_models/methane_baseline/best_reproduction' --runners 2 --pre-workers 1 --post-workers 1 --slots-per-runner 1 --iterations 0 --samples 0 --cpu-cores 4 --warmup 2 --no-pin --no-validate
