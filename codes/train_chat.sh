#!/bin/bash

PRE_SEQ_LEN=128
LR=2e-2
NUM_GPUS=2
port=$(shuf -n 1 -i 10086-65535)
export CUDA_VISIBLE_DEVICES=6,7

OMP_NUM_THREADS=12 torchrun --nnodes=1 --master_port=${port} --nproc-per-node=$NUM_GPUS main.py \
    --do_train \
    --train_file data/single-conversation/single_train_jsonl.json \
    --validation_file data/single-conversation/single_valid_jsonl.json \
    --test_file data/single-conversation/single_test_jsonl.json \
    --preprocessing_num_workers 10 \
    --prompt_column prompt \
    --response_column response \
    --history_column history \
    --overwrite_cache \
    --model_name_or_path /data1/share/edunlp/chatglm-6b-v3 \
    --output_dir runs/chatglm3-6b-socrates-single-conversation-0.25-${PRE_SEQ_LEN}-${LR}/$(date +"%Y-%m-%d_%H:%M:%S") \
    --overwrite_output_dir \
    --max_source_length 1024 \
    --max_target_length 256 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --predict_with_generate \
    --num_train_epochs 2 \
    --logging_steps 10 \
    --save_strategy epoch \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4 \
    --ptuning_checkpoint runs/chatglm3-6b-socrates-128-2e-2/2024-04-29_09:45:43/checkpoint-1000
    # --train_problem_solving_file data/problem-solving/gsm8k_train_jsonl_0.25.json \
