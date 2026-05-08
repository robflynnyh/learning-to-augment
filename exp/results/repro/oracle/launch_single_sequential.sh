#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."
mkdir -p exp/results/repro/oracle/logs

screen -L -Logfile "exp/results/repro/oracle/logs/single_sequential.log" \
  -dmS "l2a_oracle_single_sequential" \
  bash -lc './exp/results/repro/oracle/jobs/rmm_historical_then_recent.sh && ./exp/results/repro/oracle/jobs/rfm_historical_then_recent.sh'

screen -ls | grep 'l2a_'
