# SocraticLM: Exploring Socratic Personalized Teaching with Large Language Models

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/Ljyustc/SocraticLM/blob/main/LICENSE/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/Ljyustc/SocraticLM/blob/main/LICENSE/DATA_LICENSE)
[![Weight Diff License](https://img.shields.io/badge/Weight%20Diff%20License-CC%20By%20NC%204.0-yellow)](https://github.com/Ljyustc/SocraticLM/blob/main/LICENSE/WEIGHT_DIFF_LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This is the repo for paper ["SocraticLM: Exploring Socratic Personalized Teaching with Large Language Models" (NeurIPS'2024 Spotlight)](https://proceedings.neurips.cc/paper_files/paper/2024/file/9bae399d1f34b8650351c1bd3692aeae-Paper-Conference.pdf). The repo contains:

- The [SocraTeach dataset](#socrateach-dataset) used for fine-tuning SocraticLM.
- The code for [fine-tuning ChatGLM to get SocraticLM](#fine-tuning).

An enhanced implementation of our SocraticLM based on [Qwen2.5-Math-7B-Instruct](https://github.com/QwenLM/Qwen2.5-Math) can be found in our Huggingface repo [CogBase/SocraticLM](https://huggingface.co/CogBase-USTC/SocraticLM).

## Reference
If you find this repository helpful, please cite our paper.
```
@article{liu2024socraticlm,
  title={SocraticLM: exploring socratic personalized teaching with large language models},
  author={Liu, Jiayu and Huang, Zhenya and Xiao, Tong and Sha, Jing and Wu, Jinze and Liu, Qi and Wang, Shijin and Chen, Enhong},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={85693--85721},
  year={2024}
}
```

## Environment
* OS: CentOS Linux release 7.7.1908
* CPU: 15 vCPU Intel(R) Xeon(R) Platinum 8358P CPU @ 2.60GHz
* GPU: NVIDIA RTX 3090 GPUs
* CUDA: 12.1

## SocraTeach Dataset
- [`SocraTeach_multi.json`](data/SocraTeach_multi.json) is a dataset containing 35K multi-round "Teacher-Student" teaching dialogues. The keys of "SocraTeach_multi.json" are individual math problems, and the values include the corresponding "problem text", "analysis", "answer", "Step-by-step Guiding Questions", and "Teaching Dialogues". In each dialogue, "system" represents the Teacher agent's instructions, and "user" represents the Student agent's responses. The "user_type" field indicates which type of real-world student scenario the Student agent is simulating, with a total of six different types.
- [`SocraTeach_single.json`](data/SocraTeach_single.json) is a dataset containing 22K single-round "Teacher-Student" teaching dialogues. The keys correspond to four types of student responses: "irrelevant," "question," "incorrect," and "correct". In each dialogue, "history" refers to the preceding conversation, "prompt" represents the augmented student reply in the current round, and "response" denotes the expected reply from the teacher.

<strong>Note</strong>: We are continuously polishing and updating our datasets, and we welcome researchers to join us in this effort. If you find any issues with the data, we highly encourage you to submit an issue or contact us via emailing jy251198@mail.ustc.edu.cn. Your feedback is invaluable in helping us improve the datasets further.

## Fine-tuning
<strong> 0. Environment Preparation </strong> 

```bash
pip install -r requirements.txt
```

<strong> 1. Data Preprocessing </strong> 

Split the dataset into train/valid/test subsets.

```bash
cd codes
python data_preprocess.py --path ../data/SocraTeach_multi.json --split_fold ../data/data_split
```

- `path`: path to the SocraTeach dataset.
- `split_fold`: folder path to save the split train/valid/test subsets.

<strong> 2. Run the training code </strong> 

```bash
cd codes
bash train_chat.sh
```

- `train_file/validation_file/test_file`: path to your train/valid/test subsets.
- `output_dir`: path to save model checkpoint.
- `model_name_or_path`: path to the original ChatGLM3-6b weights.
- If you need to fine-tune on an existing checkpoint, please uncomment `ptuning_checkpoint` and specify the path to the checkpoint.
- If you need to fine-tune on problem-solving data, please uncomment `train_problem_solving_file` and specify the path to the problem-solving data.

We fine-tune ChatGLM3-6b with the following details:

| Details        | ChatGLM3-6b |
|----------------|------------|
| Batch size     |     64     |
| Learning rate  |    0.02    |
| Epochs         |     2      |
| GPUs           |     2      |

 <strong>Note</strong>: Our code is modified based on the ChatGLM2-6b repository (https://github.com/THUDM/ChatGLM2-6B/tree/main/ptuning). After downloading the ChatGLM3-6b weights (https://huggingface.co/THUDM/chatglm3-6b/tree/main), please replace the tokenization_chatglm.py file with the one from [`our repository`](codes/tokenization_chatglm.py).

 <strong> 3. Run the evaluation code </strong> 

```bash
cd codes
bash single_evaluate.sh
```

- Choose one evaluation task from `[conversation, single-conversation, gsm8k-solving, mawps-solving]` for the `evaluation_task`.
- Modify the `validation_file` and `test_file` accordingly.
- The `customized_output_basedir` and `customized_output_dirname` together determine the output location for the evaluation results, which will be `{customized_output_basedir}/{customized_output_dirname}`.
- The `ptuning_checkpoint` parameter specifies the path where the model checkpoint to be tested is saved. If you want to test the original results of ChatGLM3-6b, please comment out this variable.





