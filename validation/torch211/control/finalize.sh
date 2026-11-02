#!/bin/bash
set -euo pipefail
while [ ! -f /workspace/validation211/complete ]; do sleep 3; done
cd /workspace/bubbleforce
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
/workspace/train-env/bin/python /workspace/summarize.py > /workspace/summary.log 2>&1
/workspace/train-env/bin/python /workspace/collect.py
cp /workspace/summary.log /workspace/torch211-compact/summary.log
tar -czf /workspace/torch211-compact.tar.gz -C /workspace torch211-compact
cd /workspace
tar --exclude='*/wandb' -cf - validation211 source-manifest.json wheels/verification.json | gzip -1 > /workspace/torch211-results.tar.gz
sha256sum /workspace/torch211-results.tar.gz > /workspace/torch211-results.sha256
printf done > /workspace/results-ready
