#!/bin/bash
# Run one arm of queue.txt by name as its own systemd user unit ts-<name> (resumes if begun), with W workers.
#   experiments/arm.sh <name> [workers]
cd "$(dirname "$0")/.."
name=$1; w=${2:-4}
line=$(grep -E -- "--name $name( |$)" queue.txt | head -1)
[ -z "$line" ] && { echo "no arm $name in queue.txt"; exit 1; }
args=$(echo "$line" | sed -E 's/--workers [0-9]+//')
res=""; [ -f "results/match_$name.jsonl" ] && res="--resume"
systemctl --user reset-failed "ts-$name" 2>/dev/null
systemd-run --user --unit="ts-$name" --working-directory="$PWD" bash -c \
  "exec .venv/bin/python -u match.py $args --workers $w $res >> logs/$name.log 2>&1"
