#!/bin/bash
set -euo pipefail
cd /workspace/bubbleforce
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_XET_HIGH_PERFORMANCE=1
python -m venv /workspace/download-env
/workspace/download-env/bin/python -m pip install --no-cache-dir huggingface-hub > /workspace/download-install.log 2>&1
(
 /workspace/download-env/bin/hf buckets sync hf://buckets/jacknzheng/bubbleforce/restarts/decreased_momentum/decreased_momentum__checkpoint_002201 /workspace/restart > /workspace/checkpoint-download.log 2>&1
 echo $? > /workspace/checkpoint-download.exit
) &
checkpoint_pid=$!
(
 /workspace/download-env/bin/python data/cached_fineweb10B.py 12 > /workspace/data-download.log 2>&1
 echo $? > /workspace/data-download.exit
) &
data_pid=$!
python -m venv --system-site-packages /workspace/train-env
/workspace/train-env/bin/python -m pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128 > /workspace/torch-install.log 2>&1
/workspace/train-env/bin/python -m pip install --no-cache-dir -r requirements.txt > /workspace/train-install.log 2>&1
wait "$checkpoint_pid"
wait "$data_pid"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=2
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 DIST_TIMEOUT_S=300
/workspace/train-env/bin/python /workspace/bubbleforce/validation/hf-restart-002201/control/run_validation.py > /workspace/restart-validation.log 2>&1
