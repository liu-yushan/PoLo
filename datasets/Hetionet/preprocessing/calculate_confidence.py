import json
import re
import numpy as np
from datetime import datetime


with open('../graph.txt', 'r') as f:
    graph = f.readlines()
with open('../train.txt', 'r') as f:
    train = f.readlines()
with open('node_edges.json', 'r') as f:
    node_edges = json.load(f)
with open("metapaths_p3.json", 'r') as f:
    metapaths = json.load(f)
with open("metapaths_p3_inv.json", 'r') as f:
    metapaths_inv = json.load(f)
with open("compounds.json", 'r') as f:
    compounds = json.load(f)
with open("diseases.json", 'r') as f:
    diseases = json.load(f)


def get_conf_length_1(path, start_entities, rel, samples=500):
    body_support = 0
    rule_support = 0
    while body_support != samples:
        source = np.random.choice(start_entities)
        try:
            neighbors_1 = node_edges[source][path[1]][path[2]]
            neighbor_1 = np.random.choice(neighbors_1)
        except:
            neighbor_1 = ''
        if neighbor_1 != '':
            body_support += 1
            triple = source + '\t' + rel + '\t' + neighbor_1 + '\n'
            if triple in train:
                rule_support += 1
    confidence = rule_support / body_support
    return confidence


def get_conf_length_2(path, start_entities, rel, samples=500):
    body_support = 0
    rule_support = 0
    while body_support != samples:
        source = np.random.choice(start_entities)
        try:
            neighbors_1 = node_edges[source][path[1]][path[2]]
            neighbor_1 = np.random.choice(neighbors_1)
        except:
            neighbor_1 = ''
        try:
            neighbors_2 = node_edges[neighbor_1][path[3]][path[4]]
            neighbor_2 = np.random.choice(neighbors_2)
        except:
            neighbor_2 = ''
        if not (neighbor_1 == '' or neighbor_2 == ''):
            body_support += 1
            triple = source + '\t' + rel + '\t' + neighbor_2 + '\n'
            if triple in train:
                rule_support += 1
    confidence = rule_support / body_support
    return confidence


def get_conf_length_3(path, start_entities, rel, samples=500):
    body_support = 0
    rule_support = 0
    while body_support != samples:
        source = np.random.choice(start_entities)
        try:
            neighbors_1 = node_edges[source][path[1]][path[2]]
            neighbor_1 = np.random.choice(neighbors_1)
        except:
            neighbor_1 = ''
        try:
            neighbors_2 = node_edges[neighbor_1][path[3]][path[4]]
            neighbor_2 = np.random.choice(neighbors_2)
        except:
            neighbor_2 = ''
        try:
            neighbors_3 = node_edges[neighbor_2][path[5]][path[6]]
            neighbor_3 = np.random.choice(neighbors_3)
        except:
            neighbor_3 = ''
        if not (neighbor_1 == '' or neighbor_2 == '' or neighbor_3 == ''):
            body_support += 1
            triple = source + '\t' + rel + '\t' + neighbor_3 + '\n'
            if triple in train:
                rule_support += 1
    confidence = rule_support / body_support
    return confidence


samples = 5000
rule_list = []
rule_dict = dict()
counter = 0
for i in range(len(metapaths)):
    path = metapaths[i]
    path_inv = metapaths_inv[i]
    print(datetime.now().time(), counter)
    if len(path) == 3:
        conf = get_conf_length_1(path, compounds, 'CtD', samples=samples)
        conf_inv = get_conf_length_1(path_inv, diseases, '_CtD', samples=samples)
    elif len(path) == 5:
        conf = get_conf_length_2(path, compounds, 'CtD', samples=samples)
        conf_inv = get_conf_length_2(path_inv, diseases, '_CtD', samples=samples)
    elif len(path) == 7:
        conf = get_conf_length_3(path, compounds, 'CtD', samples=samples)
        conf_inv = get_conf_length_3(path_inv, diseases, '_CtD', samples=samples)
    print('1: ', conf)
    print('2: ', conf_inv)
    if (conf + conf_inv) > 0:
        confidence = (conf + conf_inv) / 2
        rule = [str(confidence), 'CtD'] + path[1::2]
        rule_list.append(rule)
    counter += 1

    rule_list = sorted(rule_list, key=lambda x: x[0], reverse=True)
    rule_dict['CtD'] = rule_list
    with open('rules_p3.txt', 'w') as f:
        json.dump(rule_dict, f)
