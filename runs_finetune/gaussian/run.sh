#!/bin/bash

# OUTPUTDIR is directory containing this run.sh script
OUTPUTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

python finetune.py \
  --model "gaussian" \
  --encoder_model_dim 64 \
  --encoder_mlp_dim 256 \
  --encoder_hidden_dim 64 \
  --batch_size 25 \
  --output_dir $OUTPUTDIR