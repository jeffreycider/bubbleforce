#!/bin/bash
set -euo pipefail
while [ ! -f /workspace/python-ready ]; do sleep 3; done
cd /workspace/bubbleforce
export PYTHONPATH=/workspace/bubbleforce PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=2
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 DIST_TIMEOUT_S=300
unset TORCHINDUCTOR_MIX_ORDER_REDUCTION
/workspace/train-env/bin/python /workspace/compiler_check.py > /workspace/validation211/compiler-check.log 2>&1
set +e
/workspace/train-env/bin/python -m pytest -q tests --junitxml=/workspace/validation211/pytest.xml > /workspace/validation211/pytest.log 2>&1
printf '%s\n' "$?" > /workspace/validation211/pytest.exit
set -e
while [ ! -f /workspace/ready ]; do sleep 3; done
/workspace/train-env/bin/python /workspace/restart.py > /workspace/validation211/restart.log 2>&1
/workspace/train-env/bin/python /workspace/fresh_training.py > /workspace/validation211/fresh-driver.log 2>&1
printf done > /workspace/validation211/complete
