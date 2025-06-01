import os
import pickle
import copy
import random
import numpy as np
import json
import torch


model_name = 'ada'
# dataset = 'FB15k-237'  # wn18rr  FB15k-237
dataset = 'FB15k-237'  # wn18rr  FB15k-237
seq_len = 16
code_num = 512



with open(data_file, 'rb') as f:
    data = pickle.load(f)
eneity_list_ada = data['eneity_list']
relation_list_ada = data['relation_list']
test_pre = data['test_pre']
valid_pre = data['valid_pre']

random.shuffle(valid_pre)
len_ = int(len(valid_pre) * 0.9)
train_pre = valid_pre[:len_]
valid_pre = valid_pre[len_:]


print(len(train_pre))
print(len(valid_pre))
print(len(test_pre))
# test_pre = np.array(test_pre)
# valid_pre = np.array(valid_pre)

# print(len(eneity_list))
# print(eneity_list)
# print(relation_list)
#
# print(test_pre.shape)
# print(valid_pre.shape)
#
# print(test_pre[-1])

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

def _read_dictionary(filename):
    d = {}
    temp_list = []
    with open(filename, 'r+') as f:
        for line in f:
            line = line.strip().split('\t')
            d[line[1]] = int(line[0])
            temp_list.append(line[1])
    return d, temp_list

def _read_fb_dic(filename):
    d = {}
    with open(filename, 'r+') as f:
        for line in f:
            line = line.strip().split('\t')
            d[line[0]] = line[1].replace('_', ' ')  # 下划线转换为空格
    return d

# codes = torch.load(code_file).numpy()
ent_dic, ent_list = _read_dictionary(data_dir+'entities.dict')
rel_dic, rel_list = _read_dictionary(data_dir+'relations.dict')

# entid2name = _read_fb_dic(data_dir+'FB15k_mid2name.txt')

def get_ent2name():
    d = {}
    with open(data_dir+'entity2text.txt', 'r+') as f:
        for line in f:
            line = line.strip().split('\t')
            d[line[0]] = line[1].split(',')[0].strip()
    return d


# entid2name = _read_fb_dic(data_dir+'FB15k_mid2name.txt')
entid2name = get_ent2name()

def gen_ent_name(entity_id):
    entid = ent_list[entity_id]
    entname = entid2name[entid]
    return entname


def gen_code_str(code_list):
    str_list = [f'[CODE{i}]' for i in code_list]
    return ' '.join(str_list)



def gen_jsondata(data, train_flag=True):
    result_list = []

    for i in range(len(data)):
        temp_json = {}
        line_data = data[i]
        h, r, t = line_data[0], line_data[1], line_data[2]
        # print(len(line_data))
        entities = line_data[3:][:20]  # 前20

        entities = list(entities)

        if train_flag and t not in entities:
        # if t not in entities:
            continue


        h = eneity_list_ada[h]
        t = eneity_list_ada[t]
        entities = [eneity_list_ada[item] for item in entities]
        # if train_flag and random.random() > 0.2:  # 训练期间  不进行一个筛选
        #     random.shuffle(entities)  # 随机打乱



        top3_ents = copy.deepcopy(entities[:3])
        if t in top3_ents:
            top3_ents.remove(t)
        t3 = [t]  # 放到前面
        t3.extend(top3_ents)



        h_name = entid2name[h]
        t_name = entid2name[t]
        if r >= len(relation_list_ada):
            r_name = 'inverse relation of ' + relation_list_ada[r-len(relation_list_ada)].replace('_', ' ').strip()
        else:
            r_name = relation_list_ada[r].replace('_', ' ').strip()


        entities_name = [entid2name[item] for item in entities]


        t3 = t3[:3]
        c_str = ''
        for lll, ent in enumerate(t3):
            codes_ = ent_dic[ent]
            # code_str = gen_code_str(codes_)
            c_str += '{}, {}\n'.format(lll + 1, codes_)
        c_str = c_str.strip()


        prompts = 'This is a knowledge graph completion task, which needs to predict the tail entity for an incomplete query triplet.\n' \
                  'The query triplet is ({}, {}, ?).\n'.format(h_name, r_name)
        h_code = codes[ent_dic[h]]
        h_code_str = gen_code_str(h_code)
        # prompts += f'The quantized representation of entity {h_name} is {h_code_str}\n'
        # prompts += 'The answer candidates and corresponding quantized representations are as follows:\n'
        prompts += 'The answer candidates are as follows:\n'
        for kkk in range(len(entities_name)):
            name = entities_name[kkk]
            # id_ = entities[i]
            id_ = ent_dic[entities[kkk]]
            # print(id_)
            codes_ = codes[id_]
            code_str = gen_code_str(codes_)
            # prompts += '{}. {}, {}\n'.format(i+1, name, code_str)
            prompts += '{}, {}\n'.format(name, code_str)
        # prompts += '\nPlease generate quantized representations of the top-3 potential answer entities, ' \
        prompts += '\nPlease generate top-3 potential answer entities, ' \
                   'ranked from highest to lowest: '
        t = ent_dic[t]
        t_code_str = entid2name[t]
        temp_json['type'] = 'kg_comp'
        temp_json['prompt'] = prompts
        temp_json['completion'] = c_str
        temp_json['answer'] = t_code_str


        result_list.append(temp_json)
    return result_list

train_json = gen_jsondata(train_pre, train_flag=True)
valid_json = gen_jsondata(valid_pre, train_flag=False)
test_json = gen_jsondata(test_pre, train_flag=False)
random.shuffle(train_json)

#
# len_ = int(len(valid_json) * 0.9)
# train_json = valid_json[:len_]
# valid_json = valid_json[len_:]

print(len(train_json))
print(len(valid_json))
print(len(test_json))


with open(out_dir+'train.json', 'w') as f:
    json.dump(train_json, f, indent=4)
with open(out_dir+'valid.json', 'w') as f:
    json.dump(valid_json, f, indent=4)
with open(out_dir+'test.json', 'w') as f:
    json.dump(test_json, f, indent=4)



