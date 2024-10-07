#!/bin/bash

run_task () {
    port=$1
    cuda_device_index=$2
    PRE_SEQ_LEN=$3
    evaluation_task=$4
    ptuning_checkpoint=$5
    customized_output_basedir=$6
    customized_output_dirname=$7

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

    if [[ -f "${output_dir}/generated_predictions.json" ]]
    then
        echo "${output_dir} already exists, please reconfirm if overwrite the results in it."
        return 0
    fi

    echo "Customized Options: ${options}"
    #CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun #--standalone --nnodes=1 --nproc-per-node=$NUM_GPUS 
    CUDA_VISIBLE_DEVICES=${cuda_device_index} torchrun --master-port=${port} main.py \
        --do_predict \
        --overwrite_cache \
        --model_name_or_path /data1/share/edunlp/chatglm-6b-v3 \
        --overwrite_output_dir \
        --max_source_length 1024 \
        --max_target_length 256 \
        --per_device_eval_batch_size 4 \
        ${options} \
        --predict_with_generate
}

PRE_SEQ_LEN=128
# export CUDA_VISIBLE_DEVICES=1
# dataset_name=gsm8k

# ptuning_checkpoint="runs/chatglm3-6b-socrates-problem-solving-128-2e-2/2024-05-10_11:11:07/checkpoint-1464"
# evaluation_task in ['gsm8k-solving', 'mawps-solving', 'single-conversation', 'conversation']
# evaluation_task="gsm8k-solving"

# customized_output_dirname="problem-solving"
checkpoints=(
    # "runs/chatglm3-6b-socrates-single-conversation-0.25-128-2e-2/2024-05-13_10:05:46/checkpoint-530"     # 0.25
    # "runs/chatglm3-6b-socrates-single-conversation-0.5-128-2e-2/2024-05-13_10:05:15/checkpoint-530"     # 0.5
    # "runs/chatglm3-6b-socrates-single-conversation-0.75-128-2e-2/2024-05-13_01:29:41/checkpoint-530"    # 0.75
    # "runs/chatglm3-6b-socrates-single-conversation-full-128-2e-2/2024-05-13_01:28:24/checkpoint-530"    # full
    "runs/chatglm3-6b-socrates-problem-solving-128-2e-2/2024-05-13_01:23:54/checkpoint-5854"
)
# suffix_name=("0.25" "0.5" "0.75" "full")
suffix_name=("0.25")
# tasks_name=("single-conversation" "conversation")
tasks_name=("gsm8k-solving" "mawps-solving")

cuda_device_idx=2

for evaluation_task in "${tasks_name[@]}"
do
    for checkpoint_idx in "${!checkpoints[@]}"
    do
        if [[ $cuda_device_idx -gt 7 ]]
        then
            break
        fi
        port=$(shuf -n 1 -i 10086-65535)
        ptuning_checkpoint=${checkpoints[$checkpoint_idx]}
        portion_suffix=${suffix_name[$checkpoint_idx]}
        customized_output_basedir="runs/chatglm3-6b-socrates-35000-problem-solving-${portion_suffix}-eval"

        run_task $port $cuda_device_idx $PRE_SEQ_LEN $evaluation_task $ptuning_checkpoint $customized_output_basedir $customized_output_dirname &

        cuda_device_idx=$(($cuda_device_idx + 1))
    done
done
