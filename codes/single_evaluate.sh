#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
port=$(shuf -n 1 -i 10086-65535)
PRE_SEQ_LEN=128
export CUDA_VISIBLE_DEVICES=0
# evaluation_task in ['conversation', 'single-conversation', 'gsm8k-solving', 'mawps-solving', ]
evaluation_task="conversation"
validation_file="$SCRIPT_DIR/../data/data_split/valid_dialogue.jsonl"
test_file="$SCRIPT_DIR/../data/data_split/test_dialogue.jsonl"
customized_output_basedir="runs/SocraticLM-eval"
customized_output_dirname="conversation"
ptuning_checkpoint="runs/SocraticLM-conversation-checkpoint"

echo "Using file path: $test_file"
if [ -f "$test_file" ]; then
  echo "Test file exists!"
else
  echo "Test file does not exist!"
fi

if [[ -n $customized_output_basedir ]]
then
    output_basedir=$customized_output_basedir
else
    if [[ -n $ptuning_checkpoint ]]
    then
        output_basedir="runs/SocraticLM-eval"
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
        validation_file="$SCRIPT_DIR/../data/data_split/valid_dialogue.jsonl"
        test_file="$SCRIPT_DIR/../data/data_split/test_dialogue.jsonl"
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
    --model_name_or_path /data2/jyliu/chatglm3-6b \
    --overwrite_output_dir \
    --max_source_length 1024 \
    --max_target_length 256 \
    --per_device_eval_batch_size 4 \
    ${options} \
    --predict_with_generate

