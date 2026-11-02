#!/bin/bash
set -euo pipefail
cd /workspace/bubbleforce
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_XET_HIGH_PERFORMANCE=1
python -m venv /workspace/download-env
/workspace/download-env/bin/python -m pip install --no-cache-dir huggingface-hub > /workspace/download-install.log 2>&1
(/workspace/download-env/bin/hf buckets sync hf://buckets/jacknzheng/bubbleforce/restarts/decreased_momentum/decreased_momentum__checkpoint_002201 /workspace/restart > /workspace/checkpoint-download.log 2>&1; echo done > /workspace/checkpoint-ready) &
checkpoint_pid=$!
(/workspace/download-env/bin/python data/cached_fineweb10B.py 12 > /workspace/data-download.log 2>&1; echo done > /workspace/data-ready) &
data_pid=$!
python -m venv --system-site-packages /workspace/train-env
/workspace/download-env/bin/python /workspace/download_wheels.py > /workspace/wheels.log 2>&1
/workspace/train-env/bin/python -m pip install --no-cache-dir /workspace/wheels/*.whl > /workspace/torch-install.log 2>&1
sed -i 's/^torch==2.10$/torch==2.11.0/' requirements.txt
/workspace/train-env/bin/python -m pip install --no-cache-dir -r requirements.txt > /workspace/train-install.log 2>&1
echo done > /workspace/python-ready
wait "$checkpoint_pid"
wait "$data_pid"
echo done > /workspace/ready
