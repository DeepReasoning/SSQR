import os
import json
import torch
import random
from  collections import defaultdict
# dataset_ = 'FB15K-237N'
dataset_ = 'CoDeX-S'


train_data_file = f'data/{dataset_}/train2id.txt'
valid_data_file = f'data/{dataset_}/valid2id.txt'
test_data_file = f'data/{dataset_}/test2id.txt'

seq_len = 32
code_num = 2048
ratio = 1  # 正负样本比例

out_dir = f'data/data4llm/{dataset_}_{seq_len}_{code_num}_r{ratio}/'
# out_dir = f'data/data4llm/{dataset_}_{seq_len}_{code_num}_nocode/'
code_file = '/raid/hpc/qika/24project/Encoder/KGENC/MyKGVQ/Class/codes_new/{}_{}_{}.pt'.format(dataset_, seq_len, code_num)

if not os.path.exists(out_dir):
    os.mkdir(out_dir)


hr2t_dic = defaultdict(list)

def load_triple(file_name):
    flag = 0
    data_ = []
    with open(file_name, 'r') as f:
        for line in f:
            if flag == 0:
                flag = 1
            else:
                h, t, r = [int(item) for item in line.strip().split(' ')]
                hr2t_dic[(h, r)].append(t)
                data_.append([h, r, t])
    return data_


def get_relname():
    rel_list = []
    with open(f'data/{dataset_}/relation2id.txt') as f:
        flag = 0
        for line in f:
            if flag == 0:
                flag += 1
                continue
            line_list = line.strip().split('\t')
            a, b = line_list
            b = int(b)
            rel_list.append(a)
    return rel_list


def get_entname():
    entityname_list = []
    ent_list = []
    with open(f'data/{dataset_}/entity2id.txt') as f:
        flag = 0
        for line in f:
            if flag == 0:
                flag += 1
                continue
            line_list = line.strip().split('\t')
            a, b = line_list
            b = int(b)
            ent_list.append(a)

    if 'FB' in dataset_:
        entid2name_dic ={}
        with open(f'data/{dataset_}/entity2text.txt') as f:
            for line in f:
                line_list = line.strip().split('\t')
                a, b = line_list
                entid2name_dic[a] = b
        for id_ in ent_list:
            entityname_list.append(entid2name_dic[id_])
    else:
        with open(f'data/{dataset_}/entities.json') as f:
            ent_json = json.load(f)
        for id_ in ent_list:
            entityname_list.append(ent_json[id_]['label'])
    return entityname_list


train_data = load_triple(train_data_file)
valid_data = load_triple(valid_data_file)
test_data = load_triple(test_data_file)
# print(hr2t_dic)

relname_list = get_relname()
entname_list = get_entname()


print(len(relname_list))
print(len(entname_list))


def get_neg_data(data_, mode='train'):
    ent_list = list(range(len(entname_list)))
    if mode == 'train':
        rrr = ratio
    else:
        rrr = 1
    new_data = []
    for item in data_:
        h, r, t = item
        true_t_list = hr2t_dic[(h, r)]
        a = random.sample(list(set(ent_list)-set(true_t_list)), rrr)
        # print(a)
        for ent in a:
            new_data.append([h, r, ent])
    return new_data


neg_train = get_neg_data(train_data)
neg_valid = get_neg_data(valid_data, 'test')
neg_test = get_neg_data(test_data, 'test')


print(len(train_data), len(neg_train))
print(len(valid_data), len(neg_valid))
print(len(test_data), len(neg_test))


codes = torch.load(code_file).numpy()


def gen_code_str(code_list):
    str_list = [f'[CODE{i}]' for i in code_list]
    return ' '.join(str_list)



def get_prompt(h,r,t):
    h_name = entname_list[h]
    r_name = relname_list[r]
    t_name = entname_list[t]

    prompt = 'Given a triple in the knowledge graph, you need to predict its validity based on the triple itself ' \
             'and entities\' quantized representations.\n'
    prompt += 'The triple is: ({}, {}, {})\n'.format(h_name, r_name, t_name)
    h_code_str = gen_code_str(codes[h])
    t_code_str = gen_code_str(codes[t])
    prompt += f'The quantized representation of entity "{h_name}" is: {h_code_str}\n'
    prompt += f'The quantized representation of entity "{t_name}" is: {t_code_str}\n'

    # prompt = 'Given a triple in the knowledge graph, you need to predict its validity based on the triple itself.\n'
    # prompt += 'The triple is: ({}, {}, {})\n'.format(h_name, r_name, t_name)

    prompt += 'Please determine the validity of the triple and response True or False.'
    return prompt


def construct_data(data, neg_data):
    data_new = []
    for item in data:
        temp_json = {}
        h, r, t = item
        temp_json['prompt'] = get_prompt(h,r,t)
        temp_json['completion'] = 'True'
        data_new.append(temp_json)
    for item in neg_data:
        temp_json = {}
        h, r, t = item
        temp_json['prompt'] = get_prompt(h,r,t)
        temp_json['completion'] = 'False'
        data_new.append(temp_json)
    return data_new


train_json = construct_data(train_data, neg_train)
valid_json = construct_data(valid_data, neg_valid)
test_json = construct_data(test_data, neg_test)

print(len(train_json))
print(len(valid_json))
print(len(test_json))
random.shuffle(train_json)

with open(out_dir+'train.json', 'w') as f:
    json.dump(train_json, f, indent=4)
with open(out_dir+'valid.json', 'w') as f:
    json.dump(valid_json, f, indent=4)
with open(out_dir+'test.json', 'w') as f:
    json.dump(test_json, f, indent=4)
