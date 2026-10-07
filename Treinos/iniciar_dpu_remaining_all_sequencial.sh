#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
run_dir="Modelos_treinados/remaining_all_dpu_retrain_original"
mkdir -p "$run_dir"
pid_file="$run_dir/runner.pid"
log_file="$run_dir/runner.log"

if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "Treino já está rodando com PID $(cat "$pid_file")"
    exit 0
fi

nohup setsid .venv/bin/python -u -m Treinos.treinar_dpu_remaining_all_sequencial \
    > "$log_file" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$pid_file"
sleep 5
if ! kill -0 "$pid" 2>/dev/null; then
    cat "$log_file"
    exit 1
fi
echo "Treinos sequenciais iniciados: PID $pid; log $log_file"
