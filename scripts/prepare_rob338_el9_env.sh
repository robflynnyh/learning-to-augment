#!/usr/bin/env bash
# Rebuild the ROB-338 evaluation environment on an EL9 compute node.

set -euo pipefail

BASE_PYTHON="${ROB338_EL9_BASE_PYTHON:-}"
CONDA_BIN="${ROB338_EL9_CONDA_BIN:-}"
CONDA_CHANNEL="${ROB338_EL9_CONDA_CHANNEL:-conda-forge}"
ENV_DIR="${ROB338_EL9_ENV_DIR:-/mnt/parscratch/users/acp21rjf/conda/rob338-el9}"
SOURCE_PYTHON="${ROB338_FREEZE_SOURCE_PYTHON:-/mnt/parscratch/users/acp21rjf/conda/main/bin/python}"
REQUIREMENTS_PATH="${ROB338_EL9_REQUIREMENTS_PATH:-${ENV_DIR}.requirements.txt}"
TORCH_INDEX_URL="${ROB338_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}"
LCASR_ROOT="${ROB338_LCASR_ROOT:-/mnt/parscratch/users/acp21rjf/long-context-asr}"
RESUME="${ROB338_EL9_RESUME:-0}"

if [ ! -r /etc/os-release ]; then
  echo "Cannot verify the compute-node operating system" >&2
  exit 1
fi
# shellcheck disable=SC1091
. /etc/os-release
if [[ "${VERSION_ID:-}" != 9* ]]; then
  echo "ROB-338 environment must be built on EL9; found VERSION_ID=${VERSION_ID:-unknown}" >&2
  exit 1
fi

if [ -n "${BASE_PYTHON}" ] && [ -n "${CONDA_BIN}" ]; then
  echo "Set only one of ROB338_EL9_BASE_PYTHON or ROB338_EL9_CONDA_BIN" >&2
  exit 1
fi
if [ -z "${BASE_PYTHON}" ] && [ -z "${CONDA_BIN}" ]; then
  echo "Set ROB338_EL9_BASE_PYTHON or ROB338_EL9_CONDA_BIN" >&2
  exit 1
fi
for required_path in "${SOURCE_PYTHON}" ${BASE_PYTHON:+"${BASE_PYTHON}"} ${CONDA_BIN:+"${CONDA_BIN}"}; do
  if [ ! -x "${required_path}" ]; then
    echo "Missing executable: ${required_path}" >&2
    exit 1
  fi
done
if [ ! -f "${LCASR_ROOT}/lcasr/__init__.py" ]; then
  echo "Missing LCASR source package: ${LCASR_ROOT}/lcasr/__init__.py" >&2
  exit 1
fi

if [ -e "${ENV_DIR}" ]; then
  if [ ! -x "${ENV_DIR}/bin/python" ]; then
    echo "Incomplete environment path already exists: ${ENV_DIR}" >&2
    exit 1
  fi
  if [ "${RESUME}" != "1" ]; then
    echo "Environment already exists: ${ENV_DIR}" >&2
    echo "Set ROB338_EL9_RESUME=1 to resume dependency installation without replacing it." >&2
    exit 1
  fi
  echo "[rob338-el9-env] resuming environment=${ENV_DIR}"
else
  mkdir -p "$(dirname "${ENV_DIR}")" "$(dirname "${REQUIREMENTS_PATH}")"
  if [ -n "${CONDA_BIN}" ]; then
    echo "[rob338-el9-env] conda_bootstrap=${CONDA_BIN}"
    echo "[rob338-el9-env] conda_channel=${CONDA_CHANNEL}"
    "${CONDA_BIN}" create \
      --yes \
      --override-channels \
      --channel "${CONDA_CHANNEL}" \
      --prefix "${ENV_DIR}" \
      python=3.10 \
      pip
  else
    echo "[rob338-el9-env] base_python=${BASE_PYTHON}"
    "${BASE_PYTHON}" -m venv "${ENV_DIR}"
  fi
fi

"${ENV_DIR}/bin/python" - <<'PY'
import platform
import sys

if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"ROB-338 expects Python 3.10, got {sys.version}")
print(f"[rob338-el9-env] python_executable={sys.executable}")
print(f"[rob338-el9-env] python={platform.python_version()}")
print(f"[rob338-el9-env] platform={platform.platform()}")
PY

"${ENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${ENV_DIR}/bin/python" -m pip install \
  torch==2.5.1+cu118 \
  torchaudio==2.5.1+cu118 \
  torchvision==0.20.1+cu118 \
  --index-url "${TORCH_INDEX_URL}"

"${SOURCE_PYTHON}" -m pip freeze | grep -Ev \
  '(^-e | @ file:|^apex @|^flash-attn==|^huggingface[_-]hub==|^torch==|^torchaudio==|^torchvision==|^triton==|^nvidia-)' \
  > "${REQUIREMENTS_PATH}"
"${ENV_DIR}/bin/python" -m pip install -r "${REQUIREMENTS_PATH}"

PYTHONPATH="/mnt/parscratch/users/acp21rjf/symphony-workspaces-learning-to-augment/ROB-338:/mnt/parscratch/users/acp21rjf/symphony-workspaces-learning-to-augment/ROB-338/exp:${LCASR_ROOT}:/mnt/parscratch/users/acp21rjf/language_modelling" \
  "${ENV_DIR}/bin/python" - <<'PY'
import numpy
import omegaconf
import torch
import torchaudio
import yaml

import l2augment
import lcasr

print(f"[rob338-el9-env] torch={torch.__version__}")
print(f"[rob338-el9-env] torchaudio={torchaudio.__version__}")
print(f"[rob338-el9-env] numpy={numpy.__version__}")
print(f"[rob338-el9-env] yaml={yaml.__version__}")
print(f"[rob338-el9-env] cuda_build={torch.version.cuda}")
print(f"[rob338-el9-env] lcasr_source={lcasr.__file__}")
print("[rob338-el9-env] import_validation=passed")
PY

echo "[rob338-el9-env] environment=${ENV_DIR}"
echo "[rob338-el9-env] requirements=${REQUIREMENTS_PATH}"
