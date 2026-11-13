#!/usr/bin/env bash
# Start, watch or stop a training run on the shared GPU node. No scheduler: it runs in the background with nohup.
#   GPU=5 bash train/run.sh start NAME [train.py arguments...]     e.g. start h1b-lora-s0 --model ibm-granite/granite-4.0-h-1b --data data/generated --out runs/h1b-lora-s0
#   bash train/run.sh status NAME      last log lines and whether it is alive
#   bash train/run.sh stop NAME
# GPU 6 is off limits (the owner's rule on this node); the script refuses it.
set -u
cmd=${1:-}; name=${2:-}; shift 2 2>/dev/null || true
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs; log=runs/$name.log; pidf=runs/$name.pid
case "$cmd" in
  start)
    gpu=${GPU:?set GPU=<index> (never 6)}
    [ "$gpu" = "6" ] && { echo "GPU 6 is off limits" >&2; exit 2; }
    CUDA_VISIBLE_DEVICES=$gpu nohup python train/train.py "$@" > "$log" 2>&1 &
    echo $! > "$pidf"; echo "started $name pid $(cat "$pidf") on GPU $gpu, log $log" ;;
  status)
    if [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; then echo "$name: running (pid $(cat "$pidf"))"; else echo "$name: not running"; fi
    tail -n 15 "$log" ;;
  stop)
    [ -f "$pidf" ] && kill "$(cat "$pidf")" && echo "stopped $name" ;;
  *) echo "usage: GPU=N bash train/run.sh start|status|stop NAME [args]" >&2; exit 1 ;;
esac
