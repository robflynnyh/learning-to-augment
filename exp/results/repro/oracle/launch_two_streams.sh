#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."
mkdir -p exp/results/repro/oracle/logs

for method in rmm rfm; do
  job="exp/results/repro/oracle/jobs/${method}_historical_then_recent.sh"
  screen -L -Logfile "exp/results/repro/oracle/logs/${method}_historical_then_recent.log" \
    -dmS "l2a_${method}_historical_then_recent" bash "${job}"
done

screen -ls | grep 'l2a_'
