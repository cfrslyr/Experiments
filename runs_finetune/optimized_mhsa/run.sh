#!/bin/bash

# OUTPUTDIR is directory containing this run.sh script
OUTPUTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

python finetune.py \
  --model "optimized_mhsa" \
  --load_checkpoint_file "runs_pretrain/optimized_mhsa/final.checkpoint" \
  --output_dir $OUTPUTDIR