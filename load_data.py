#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  File name:    load_data.py
"""

import time
from collections import defaultdict
from copy import deepcopy
import numpy as np
import re
import os
import itertools
import utils
import random
import string

def load_phrase_dic(fn):
    dic = defaultdict()
    with open(fn, 'r', encoding="utf-8") as f:
        for ln in f.read().strip().split('\n'):
            k, v = ln.split(":")
            k = k.strip()
            dic[k] = v.strip()
    return dic


def load_babi_dic(fn):
    dic = defaultdict()
    with open(fn, 'r', encoding="utf-8") as f:
        for ln in f.read().strip().split('\n'):
            ln = ln.lower()
            k, v = ln.split()
            dic[k] = v
    return dic


class SyllogismData:
    def __init__(self, data_dir="data/Syllogism/", use_negation=0, use_random_symbol=False):
        t_ = time.time()
        self.data_dir = data_dir
        self.filenames = ["syllogism4pretrained.txt"]
        self.raw_data_list = []
        raw_data_list = [self.load_syllogism(data_dir + filename) for filename in self.filenames]
        table = {}
        i = 0
        min_len = 100000
        for lst in raw_data_list:
            table[i] = lst
            i += 1
            if len(lst) < min_len:
                min_len = len(lst)
        for j in range(min_len):
            for k in range(i):
                self.raw_data_list.append(table[k][j])

        self.id2ent_dict_list = []
        self.data_list = self.process(self.raw_data_list)
        self.target = [[True, False]]
        self.use_negation = use_negation
        self.use_random_symbol = use_random_symbol

        self.init_time = time.time() - t_

    def load_syllogism(self, file_dir, use_random_symbol=False):
        with open(file_dir, "r", encoding="utf-8") as f:
            data = f.read().strip().split("\n")
            if use_random_symbol:
                data = [self.replace_random_symbols(re.split(", |: |; ", d)) for d in data]
            else:
                data = [re.split(", |: |; ", d) for d in data]
        return data

    def replace_random_symbols(self, lst):
        rdict = dict()
        for rs in ["S", "M0", "P"]:
            r1 = ''.join([random.choice(string.ascii_letters) for n in range(6)])
            r2 = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(6)])
            rdict[rs] = r1+r2
        return [" ".join([rdict.get(e, e) for e in rd.split()]) for rd in lst]

    def process(self, data):
        new_data = []
        tmp_data_arr = []
        for d in data:
            ent_list = [d[-1].split(" ")[1]]
            tmp_data_item = []
            for s in d:
                ws = s.split()
                if ws[1] not in ent_list:
                    ent_list.append(ws[1])
                if ws[2] not in ent_list:
                    ent_list.append(ws[2])
                tmp_data_item.append([ws[1], ws[2], utils.spatialise_relation(ws[0])])
            ent2id_dict = dict((n, i) for i, n in enumerate(ent_list))
            self.id2ent_dict_list.append(dict((i, n) for i, n in enumerate(ent_list)))
            tmp_data_arr.append([tmp_data_item, ent2id_dict])
        
        for (item, d) in tmp_data_arr:
            new_data.append([len(d), [(d[j[0]], d[j[1]], j[2], None) for j in item]])
        return new_data

    def __repr__(self):
        pass
        return self.__class__.__name__ + " dataset(" + self.data_dir + ") summary:" + \
            "\n\tfamily_num: " + str(len(self.data_list)) + \
            "\n\t----------------------------- init_time: " + str(round(self.init_time, 3)) + "s"


class ValidSyllogism(SyllogismData):

    def __init__(self, data_dir="data/ValidSyllogism", file_name = "Possible_Syllogism_3.txt", len_premise=2, use_negation=0, use_random_symbol=False):
        self.possible_rel = ['all', 'some-not', 'no', 'some']
        self.neg_dic = {"all":"some-not", "some-not":"all", "no":"some", "some":"no"}
        t_ =time.time()
        self.data_dir = data_dir
        if 'ValidSyllogism' in data_dir: # for experiment 1
            self.filenames = [file_name]
        else:                         # for experiment 3
            self.filenames = [
                "syllogism_3.txt",  # 1 : 120
                "syllogism_4.txt",  # 2 : 120
                "syllogism_5.txt",  # 3,4 : 120
                "syllogism_6.txt",  # 5,6 : 120
                "syllogism_7.txt",  # 7,8 : 120
                "syllogism_8.txt",  # 9 : 120
                "syllogism_9.txt",  # 10,11 : 120
                "syllogism_10.txt",  # 12 : 120
                "syllogism_11.txt",  # 13,14 : 120
                "syllogism_12.txt",  # 15,16 : 120
        ]
        for fn in self.filenames:
            if not os.path.exists('/'.join([data_dir,fn])):
                file_name = "{}/{}_{}.txt".format(data_dir, "Possible_Syllogism", len_premise)
                self.create_possible_syllogism(file_name, len_premise=len_premise)

        data_dir = data_dir.strip('/')
        raw_data_list = [self.load_syllogism('/'.join([data_dir, filename]), use_random_symbol=use_random_symbol) for filename in self.filenames]
        self.raw_data_list = [j for i in raw_data_list for j in i]
        self.id2ent_dict_list = []
        self.data_list = self.process(self.raw_data_list)

        self.use_negation = use_negation
        self.init_time = time.time() - t_

    def create_possible_syllogism(self, file_name, len_premise=2):
        Entities = ["S"] + ["M{}".format(e) for e in range(len_premise - 1)] + ["P"]
        Pairs = list(zip(Entities[:-1],Entities[1:]))
        Permutated_Pairs = list(itertools.product(*map(lambda x: list(itertools.permutations(x)), Pairs)))
        PossibleRelations = list(itertools.product(*[self.possible_rel]*len(Pairs)))
        self.possible_premises = [list(zip(rels, pairs)) for rels in PossibleRelations for pairs in Permutated_Pairs]
        self.possible_premises = [[[p[0]]+list(p[1]) for p in premises] for premises in self.possible_premises]
        #print(self.possible_premises)
        #print(self.possible_conclusions)
        possible_conclusions = [[c[0]] + c[1] for c in list(zip(self.possible_rel, [["S", "P"]] * 4))]
        all_possibilities = [s[0]+[s[1]] for s in itertools.product(self.possible_premises, possible_conclusions)]
        print(len(all_possibilities))
        with open(file_name, 'w') as fh:
            lines = '\n'.join([', '.join([' '.join(lst) for lst in plst]) for plst in all_possibilities])
            fh.write(lines)
        return file_name


class ChooseValidSyllogism(ValidSyllogism):
    def __init__(self, data_dir="data/multiple_choice/", data="", use_negation=0,use_random_symbol=False):
        t_ = time.time()
        self.data_dir = data_dir
        self.filenames = [
            "syllogism_3.txt",  # 1 : 120
            "syllogism_4.txt",  # 2 : 120
            "syllogism_5.txt",  # 3,4 : 120
            "syllogism_6.txt",  # 5,6 : 120
            "syllogism_7.txt",  # 7,8 : 120
            "syllogism_8.txt",  # 9 : 120
            "syllogism_9.txt",  # 10,11 : 120
            "syllogism_10.txt",  # 12 : 120
            "syllogism_11.txt",  # 13,14 : 120
            "syllogism_12.txt",  # 15,16 : 120
        ]
        if data != "":
            self.filenames = [data]
        self.raw_data_list = []
        raw_data_list = [self.load_syllogism(data_dir + filename, use_random_symbol=use_random_symbol) for filename in self.filenames]
        table = {}
        i = 0
        min_len = 100000
        for lst in raw_data_list:
            table[i] = lst
            i += 1
            if len(lst) < min_len:
                min_len = len(lst)
        for j in range(min_len):
            for k in range(i):
                self.raw_data_list.append(table[k][j])
                # print(table[k][j])

        #        self.raw_data_list = [j for i in raw_data_list for j in i]
        self.id2ent_dict_list = []
        self.data_list = self.process(self.raw_data_list)
        self.target = [[True, False]]
        self.use_negation = use_negation

        self.init_time = time.time() - t_

    def process(self, data):
        new_data = []
        tmp_data_arr = []
        for d in data:
            ent_list = [d[-1].split(" ")[1]]
            tmp_data_item = []
            for s in d:
                ws = s.split(" ")
                if ws[1] not in ent_list:
                    ent_list.append(ws[1])
                if ws[2] not in ent_list:
                    ent_list.append(ws[2])
                tmp_data_item.append([ws[1], ws[2], utils.spatialise_relation(ws[0])])
            ent2id_dict = dict((n, i) for i, n in enumerate(ent_list))
            self.id2ent_dict_list.append(dict((i, n) for i, n in enumerate(ent_list)))
            tmp_data_arr.append([tmp_data_item, ent2id_dict])

        for (item, d) in tmp_data_arr:
            new_data.append([len(d), [(d[j[0]], d[j[1]], j[2], None) for j in item]])
        return new_data

    def __repr__(self):
        pass
        return self.__class__.__name__ + " dataset(" + self.data_dir + ") summary:" + \
               "\n\tfamily_num: " + str(len(self.data_list)) + \
               "\n\t----------------------------- init_time: " + str(round(self.init_time, 3)) + "s"


class PowerSphere(SyllogismData):
    def __init__(self, data_dir="data/Syllogism/", use_negation=0):
        t_ = time.time()
        self.data_dir = data_dir
        self.filename = "List_of_valid_deduction.txt"
        self.raw_data_list = self.load_syllogism(data_dir + self.filename)
        self.id2ent_dict_list = []
        self.data_list = self.process(self.raw_data_list)
        #self.data_list, self.func_list = self.process(self.raw_data_list)
        self.target = [[True, False]]
        self.use_negation = use_negation

        self.init_time = time.time() - t_

    def process(self, data):
        new_data = []
        tmp_data_arr = []
        for d in data:
            entu_list = []
            ents_list = []
            tmp_data_item = []
            func_dic = defaultdict(list)
            for s in d:
                ulst, slst, ftable, ptable = spatailising_statement(s)
                entu_list += ulst
                ents_list += slst
                for ws in ptable.values():
                    if ws not in tmp_data_item:
                        tmp_data_item.append(ws)
                func_dic.update(ftable)
            ent_list = list(set(entu_list)) + list(set(ents_list).difference(set(entu_list)))
            ent2id_dict = dict((n, i) for i, n in enumerate(ent_list))
            self.id2ent_dict_list.append(dict((i, n) for i, n in enumerate(ent_list)))
            tmp_data_arr.append([len(set(entu_list)), tmp_data_item, ent2id_dict, func_dic])

        for (n, item, d, fd) in tmp_data_arr:
            new_data.append([n, len(d),
                             [(d.get(j[0], -1), d.get(j[1], -1), j[len(j)-1], None) for j in item],
                             [(d.get(kv[0]), d.get(kv[1][0], -1), d.get(kv[1][1], -1), kv[1][len(kv[1])-1]) for kv in fd.items() if kv[0] in d.keys()]
                             ])
        return new_data


if __name__ == '__main__':
    # TEST
    d = SyllogismData()
    print('done!')
