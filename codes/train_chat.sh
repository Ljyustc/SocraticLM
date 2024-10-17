#!/bin/bash

PRE_SEQ_LEN=128
TYPE=Dialogue
LR=2e-2
NUM_GPUS=2
port=$(shuf -n 1 -i 10086-65535)
export CUDA_VISIBLE_DEVICES=0,1

OMP_NUM_THREADS=12 torchrun --nnodes=1 --master_port=${port} --nproc-per-node=$NUM_GPUS main.py \
    --do_train \
    --train_file ../data/data_split/train_dialogue.jsonl \
    --validation_file ../data/data_split/valid_dialogue.jsonl \
    --test_file ../data/data_split/test_dialogue.jsonl \
    --preprocessing_num_workers 10 \
    --prompt_column prompt \
    --response_column response \
    --history_column history \
    --overwrite_cache \
    --model_name_or_path /data/chatglm3-6b \
    --output_dir runs/SocraticLM-${TYPE}-${PRE_SEQ_LEN}-${LR}/$(date +"%Y-%m-%d_%H:%M:%S") \
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
    # --quantization_bit 4 \
    # --ptuning_checkpoint runs/checkpoint_name
    # --train_problem_solving_file data/problem-solving/gsm8k_train_jsonl_0.25.json \
