#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."
mkdir -p exp/results/repro/oracle/logs

for job in exp/results/repro/oracle/jobs/*.sh; do
  name="$(basename "${job}" .sh)"
  screen -L -Logfile "exp/results/repro/oracle/logs/${name}.log" \
    -dmS "l2a_${name}" bash "${job}"
done

screen -ls | grep 'l2a_'
