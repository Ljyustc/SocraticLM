import json
import random
import os

with open(r"path_to_SocraTeach_multi.json", 'r', encoding='utf-8') as f:
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

script_dir = os.path.dirname(os.path.abspath(__file__))
data_split_dir = os.path.join(script_dir, '..', 'data', 'data_split')
data_split_dir = os.path.abspath(data_split_dir)
os.makedirs(data_split_dir, exist_ok=True)

train_file_path = os.path.join(data_split_dir, "train_dialogue.json")  
valid_file_path = os.path.join(data_split_dir, "valid_dialogue.json") 
test_file_path = os.path.join(data_split_dir, "test_dialogue.json") 

with open(train_file_path, 'w', encoding='utf-8') as f:
    json.dump(train_data, f, indent=4)

with open(valid_file_path, 'w', encoding='utf-8') as f:
    json.dump(valid_data, f, indent=4)

with open(test_file_path, 'w', encoding='utf-8') as f:
    json.dump(test_data, f, indent=4)