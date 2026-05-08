#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

./exp/results/repro/oracle/RMM/run_cpu.sh
./exp/results/repro/oracle/RFM/run_cpu.sh
