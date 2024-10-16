#!/bin/bash

port=$(shuf -n 1 -i 10086-65535)
PRE_SEQ_LEN=128
export CUDA_VISIBLE_DEVICES=0
# evaluation_task in ['gsm8k-solving', 'mawps-solving', 'single-conversation', 'conversation']
evaluation_task="gsm8k-solving"
validation_file="data/problem25_single_additional_test_jsonl.json"
test_file="data/problem25_single_additional_test_jsonl.json"
customized_output_basedir="runs/chatglm3-6b-socrates-problem-solving-0.25-eval"
customized_output_dirname="problem25_single_additional_test"
ptuning_checkpoint="runs/chatglm3-6b-socrates-problem-solving-128-2e-2/2024-05-11_15:04:39/checkpoint-764"

if [[ -n $customized_output_basedir ]]
then
    output_basedir=$customized_output_basedir
else
    if [[ -n $ptuning_checkpoint ]]
    then
        output_basedir="runs/chatglm3-6b-socrates-eval"
    else
        output_basedir="runs/chatglm3-6b-eval"
    fi
fi


if [[ -n $validation_file && -n $test_file ]]
then
    prompt_column="prompt"
    response_column="response"
    history_column="history"
else
    if [[ $evaluation_task == "gsm8k-solving" ]]
    then
        validation_file="data/gsm8k_jsonl.json"
        test_file=$validation_file
        prompt_column="question"
        response_column="answer"
    elif [[ $evaluation_task == "mawps-solving" ]]
    then
        validation_file="data/mawps_jsonl.json"
        test_file=$validation_file
        prompt_column="original_text"
        response_column="original_text"
    elif [[ $evaluation_task == "single-conversation" ]]
    then
        validation_file="data/single-conversation/single_valid_jsonl.json"
        test_file="data/single-conversation/single_test_jsonl.json"
        # validation_file="data/single-conversation/single_addtional_test_jsonl.json"
        # test_file="data/single-conversation/single_addtional_test_jsonl.json"
        prompt_column="prompt"
        response_column="response"
        history_column="history"
    elif [[ $evaluation_task == "conversation" ]]
    then
        validation_file="data/valid_dialogue_jsonl.json"
        test_file="data/test_dialogue_jsonl.json"
        prompt_column="prompt"
        response_column="response"
        history_column="history"
    else
        validation_file=""
        test_file=""
        prompt_column=""
        response_column=""
    fi
fi

options="--validation_file ${validation_file} --test_file ${test_file} --prompt_column ${prompt_column} --response_column ${response_column}"

if [[ -n $ptuning_checkpoint ]]
then
    options="${options} --ptuning_checkpoint ${ptuning_checkpoint} --pre_seq_len ${PRE_SEQ_LEN}"
fi

if [[ -n $history_column ]]
then
    options="${options} --history_column ${history_column}"
fi

if [[ -n $customized_output_dirname ]]
then
    output_dir="${output_basedir}/${customized_output_dirname}"
    options="${options} --output_dir ${output_basedir}/${customized_output_dirname}"
else
    output_dir="${output_basedir}/${evaluation_task}"
    options="${options} --output_dir ${output_basedir}/${evaluation_task}"
fi

# if [[ -d ${output_dir} ]]
# then
#     echo "${output_dir} already exists, please reconfirm if overwrite the results in it."
#     return 0
# fi

echo "Customized Options: ${options}"
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun #--standalone --nnodes=1 --nproc-per-node=$NUM_GPUS 
torchrun --master-port=${port} main.py \
    --do_predict \
    --overwrite_cache \
    --model_name_or_path /data1/share/edunlp/chatglm-6b-v3 \
    --overwrite_output_dir \
    --max_source_length 1024 \
    --max_target_length 256 \
    --per_device_eval_batch_size 4 \
    ${options} \
    --predict_with_generate

# export CUDA_VISIBLE_DEVICES=1
# dataset_name=gsm8k

# ptuning_checkpoint="runs/chatglm3-6b-socrates-problem-solving-128-2e-2/2024-05-10_11:11:07/checkpoint-1464"
# evaluation_task="gsm8k-solving"

