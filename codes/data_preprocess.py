import json
import random
import os
import tqdm
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='Data preprocess')
    parser.add_argument('--path', type=str, default='data', help='Path to the SocraTeach dataset')
    parser.add_argument('--split_fold', type=str, default='data/data_split', help='Folder path to save the split train/valid/test subsets')
    args = parser.parse_args()
    return args
args = get_args()
os.makedirs(args.split_fold, exist_ok=True)

with open(args.path, 'r', encoding='utf-8') as f:
    dialog_data = json.load(f)

all_data = {}
prompt0 = "You are a Socratic teacher, please guide me to solve the [Problem] with heuristic questions based on the following information. \n[Problem]"
for d in dialog_data:
    data = dialog_data[d]
    ques = data['question']
    ana = data['analysis']
    ans = data['answer']
    promptd = prompt0 + ques + " [Answer] " + ans + " [Analysis] " + ana
    ques1 = data['steps'][0]
    
    for dia_id in data['dialogues']:
        if "END" not in data['dialogues'][dia_id][-1]:
            continue
        dia_data = [[promptd]]
        
        for con_id in range(len(data['dialogues'][dia_id])):
            con = data['dialogues'][dia_id][con_id]
            dia_data[-1].append(con['system'])
            
            all_data[dia_id+"_"+str(con_id)] = {"prompt": dia_data[-1][0], "response": dia_data[-1][1], "history": dia_data[:-1]}
            
            if 'user' in con:
                dia_data = dia_data + [[con['user']]]

keys = list(all_data.keys())
random.shuffle(keys)

test_data = {}
valid_data = {}
train_data = {}
for i in keys[:100]:
    test_data[i] = all_data[i]
for i in keys[1000:2000]:
    valid_data[i] = all_data[i]
skip_ids = set(['_'.join(x.split('_')[:3]) for x in list(test_data.keys())+list(valid_data.keys())])
for i in keys[2000:]:
    filter_i = '_'.join(i.split('_')[:3])
    if filter_i not in skip_ids:
        train_data[i] = all_data[i]

train_file_path = os.path.join(args.split_fold, "train_dialogue.jsonl")  
valid_file_path = os.path.join(args.split_fold, "valid_dialogue.jsonl") 
test_file_path = os.path.join(args.split_fold, "test_dialogue.jsonl") 

train_datal = [{"id": key, **val} for key, val in train_data.items()]
with open(train_file_path, "w") as f:
    for item in tqdm.tqdm(train_datal, ncols=128):
        f.write(json.dumps(item) + "\n")

valid_datal = [{"id": key, **val} for key, val in valid_data.items()]
with open(valid_file_path, "w") as f:
    for item in tqdm.tqdm(valid_datal, ncols=128):
        f.write(json.dumps(item) + "\n")

test_datal = [{"id": key, **val} for key, val in test_data.items()]
with open(test_file_path, "w") as f:
    for item in tqdm.tqdm(test_datal, ncols=128):
        f.write(json.dumps(item) + "\n")