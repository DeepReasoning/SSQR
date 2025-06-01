from collections import defaultdict as ddict
import os
import numpy as np


def process(dataset, num_rel):
    """
    pre-process dataset
    :param dataset: a dictionary containing 'train', 'valid' and 'test' data.
    :param num_rel: relation number
    :return:
    """
    sr2o = ddict(set)
    for subj, rel, obj in dataset['train']:
        sr2o[(subj, rel)].add(obj)
        sr2o[(obj, rel + num_rel)].add(subj)
    sr2o_train = {k: list(v) for k, v in sr2o.items()}
    for split in ['valid', 'test']:
        for subj, rel, obj in dataset[split]:
            sr2o[(subj, rel)].add(obj)
            sr2o[(obj, rel + num_rel)].add(subj)
    sr2o_all = {k: list(v) for k, v in sr2o.items()}
    triplets = ddict(list)

    for (subj, rel), obj in sr2o_train.items():
        triplets['train'].append({'triple': (subj, rel, -1), 'label': sr2o_train[(subj, rel)]})
    for split in ['valid', 'test']:
        for subj, rel, obj in dataset[split]:
            triplets[f"{split}_tail"].append({'triple': (subj, rel, obj), 'label': sr2o_all[(subj, rel)]})
            triplets[f"{split}_head"].append(
                {'triple': (obj, rel + num_rel, subj), 'label': sr2o_all[(obj, rel + num_rel)]})
    triplets = dict(triplets)
    return triplets


def _read_dictionary(filename):
    d = {}
    temp_list = []
    with open(filename, 'r+') as f:
        for line in f:
            line = line.strip().split('\t')
            d[line[1]] = int(line[0])
            temp_list.append(line[1])
    return d, temp_list

def _read_triplets(filename):
    with open(filename, 'r+') as f:
        for line in f:
            processed_line = line.strip().split('\t')
            yield processed_line

def _read_triplets_as_list(filename, entity_dict, relation_dict):
    l = []
    for triplet in _read_triplets(filename):
        s = entity_dict[triplet[0]]
        r = relation_dict[triplet[1]]
        o = entity_dict[triplet[2]]
        l.append([s, r, o])
    return l

class Data(object):
    def __init__(self, num_nodes, train_data, valid_data, test_data, num_rels):
        super(Data, self).__init__()
        self.num_nodes = num_nodes
        self.train = train_data
        self.valid = valid_data
        self.test = test_data
        self.num_rels = num_rels

def load_data(dataset_dir):
    entity_path = os.path.join(dataset_dir, 'entities.dict')
    relation_path = os.path.join(dataset_dir, 'relations.dict')
    train_path = os.path.join(dataset_dir, 'train.txt')
    valid_path = os.path.join(dataset_dir, 'valid.txt')
    test_path = os.path.join(dataset_dir, 'test.txt')
    entity_dict,entity_list = _read_dictionary(entity_path)
    relation_dict,relation_list = _read_dictionary(relation_path)
    train = np.array(_read_triplets_as_list(train_path, entity_dict, relation_dict))
    valid = np.array(_read_triplets_as_list(valid_path, entity_dict, relation_dict))
    test = np.array(_read_triplets_as_list(test_path, entity_dict, relation_dict))
    num_nodes = len(entity_dict)
    print("# entities: {}".format(num_nodes))
    num_rels = len(relation_dict)
    print("# relations: {}".format(num_rels))
    print("# edges: {}".format(len(train)))
    data = Data(num_nodes, train, valid, test, num_rels)
    return data,entity_dict,relation_dict,entity_list,relation_list



def _read_triplets_new(filename):
    flag = 0
    new_list = []
    with open(filename, 'r') as f:
        for line in f:
            if flag == 0:
                flag += 1
                continue
            line_list = line.strip().split(' ')
            line_list = [int(item) for item in line_list]
            new_list.append([line_list[0], line_list[2], line_list[1]])
            flag += 1
    return new_list


def load_data_new(dataset_dir):
    train_path = os.path.join(dataset_dir, 'train2id.txt')
    valid_path = os.path.join(dataset_dir, 'valid2id.txt')
    test_path = os.path.join(dataset_dir, 'test2id.txt')

    train = np.array(_read_triplets_new(train_path))
    valid = np.array(_read_triplets_new(valid_path))
    test = np.array(_read_triplets_new(test_path))

    if 'fb15k' in dataset_dir.lower():
        num_rels = 237
        num_nodes = 14541
    else:
        num_rels = 42
        num_nodes = 2034
    data = Data(num_nodes, train, valid, test, num_rels)
    print(num_nodes, num_rels, len(train), len(valid), len(test))
    return data
