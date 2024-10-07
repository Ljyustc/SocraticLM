# SocraticLM: Exploring Socratic Personalized Teaching with Large Language Models

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/Ljyustc/SocraticLM/blob/main/LICENSE/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/Ljyustc/SocraticLM/blob/main/LICENSE/DATA_LICENSE)
[![Weight Diff License](https://img.shields.io/badge/Weight%20Diff%20License-CC%20By%20NC%204.0-yellow)](https://github.com/Ljyustc/SocraticLM/blob/main/LICENSE/WEIGHT_DIFF_LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This is the repo for paper "SocraticLM: Exploring Socratic Personalized Teaching with Large Language Models" (NeurIPS'2024 Spotlight). The repo contains:

- The [SocraTeach dataset](#socrateach-dataset) used for fine-tuning SocraticLM.
- The code for [fine-tuning SocraticLM](#fine-tuning).
- The code for [recovering SocraticLM weights from our released weight diff](#recovering-socraticlm-weights).

## Environment
* OS: CentOS Linux release 7.7.1908
* CPU: 15 vCPU Intel(R) Xeon(R) Platinum 8358P CPU @ 2.60GHz
* GPU: NVIDIA RTX 3090 GPUs
* CUDA: 11.1

## SocraTeach Dataset
- [`SocraTeach_multi.json`](data/SocraTeach_multi.json) is a dataset containing 35K multi-round "Teacher-Student" teaching dialogues. The keys of "SocraTeach_multi.json" are individual math problems, and the values include the corresponding "problem text", "analysis", "answer", "Step-by-step Guiding Questions", and "Teaching Dialogues". In each dialogue, "system" represents the Teacher agent's instructions, and "user" represents the Student agent's responses. The "user_type" field indicates which type of real-world student scenario the Student agent is simulating, with a total of six different types.

## Fine-tuning
We fine-tune ChatGLM-8B with the following details:

| Details        | ChatGLM-8B |
|----------------|------------|
| Batch size     | 64         |
| Learning rate  | 0.02       |
| Epochs         | 2          |
| GPUs           | 2          |

## Recovering SocraticLM Weights





