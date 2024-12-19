#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  File name:    utils.py
"""

import os
import numpy as np
import spacy
import itertools
import random
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Circle
from matplotlib.lines import Line2D
import matplotlib.font_manager as font_manager

from adjustText import adjust_text

nlp = spacy.load("en_core_web_sm")


def create_list_of_sat_cases_exp2(inFile1 = '', inFile2 = '', outFile = ''):
    '''
    all three files use absolute file names.
    :param inFile2: input file of all test cases, each line is a syllogistic reasoning
    :param inFile1: input file of all valid syllogistic reasoning
    :param outFile: output a list of file numbers, n, log_text_<n>.txt should project True conclustion
    :return: the length of the list in outFile
    '''
    unsat_formats = []
    rep_dic = {'all': 'some-not', 'some': 'no', 'some-not': 'all', 'no': 'some'}
    with open(inFile1, 'r') as ifh:
        for ln in ifh.readlines():
            p1, p2, c = ln.strip().split(', ')
            cRel = c.split()[0]
            c = c.replace(cRel, rep_dic[cRel])
            unsat_formats.append(p1 + ', '+ p2 + ': ' +c)
    test_with_correct_ans = []
    rep_dic = {'all': 'P', 'some': '-D', 'some-not': '-P', 'no': 'D'}
    with open(inFile2, 'r') as ifh:
        for ln in ifh.readlines():
            ln = ln.strip()
            S, P = ln.split(':')[-1].split(' ')[2:]
            M1 = [ele for ele in ln.split(',')[0].split(' ')[1:] if ele != S][0]
            newln = ln.replace(S, 'S')
            P = P.replace(S, 'S')
            M1 = M1.replace(S, 'S')
            newln = newln.replace(P, 'P')
            M1 = M1.replace(P, 'P')
            newln = newln.replace(M1, 'M1')
            if newln in unsat_formats:
                test_with_correct_ans.append(ln + '*' + 'UNSAT')
            else:
                test_with_correct_ans.append(ln + '*' + 'SAT')
    with open(outFile, 'w+') as ofh:
        ofh.write('\n'.join(test_with_correct_ans))
    num = len(test_with_correct_ans)
    print(num)
    return num


def CREAte_babi_tagged_words(all_tasks):
    """
    :param task_list:
    :param create:
    :return:
    """
    appeared_words = []
    all_words = defaultdict(list)
    nlp = spacy.load("en_core_web_sm")
    for task in [tasks for one_type_tasks in all_tasks for tasks in one_type_tasks]:
        struct = {}
        for ln in task:
            words = [w for w in ln.split() if w not in appeared_words]
            if len(words) == 0:
                continue

            appeared_words += words
            q_a = None
            if ln.startswith("DES: "):
                ln = ln.strip("DES: ")  # "1 John went to the office"
                num = int(ln.split()[0])   # num = 1
                struct[num] = {}
                snt = nlp(" ".join(ln.split()[1:])) # snt = "John went to the office"
                struct[num][0] = "DES"
                i = 1
                for token in snt:
                    wd = {"word": token.text,
                          "pos": token.pos_,
                          "tag": token.tag_
                          }
                    struct[num][i] = wd
                    i += 1
                    all_words[token.pos_+token.tag_].append(token.text)

            elif ln.startswith("QUES: "):
                ln = ln.strip("QUES: ")
                num = int(ln.split()[0])
                struct[num] = {}
                q_a = " ".join(ln.split()[1:]).split("ANS: ")
                snt = nlp(q_a[0])
                struct[num][0] = "QUES"
                i = 1
                for token in snt:
                    wd = {"word": token.text,
                          "pos": token.pos_,
                          "tag": token.tag_
                          }
                    struct[num][i] = wd
                    i += 1
                struct[num][i] = q_a[1]
                all_words[token.pos_ + token.tag_].append(token.text)

    for k in all_words.keys():
        all_words[k] = list(set(all_words[k]))
        with open('data/bAbI/tagged_words.txt', 'a+') as fh:
            fh.write("\n".join([k]+all_words[k]))

def get_uniform_locations_on_sphere(dim=2, centra = 0, r=1, num = 10):
    rlt, count = [], 0
    while len(rlt) < num and count < num*2:
        u = np.random.normal(0, 1, dim)
        norm = np.sum(u ** 2) ** (0.5)
        u = u*r/norm
        u += centra
        if len([e for e in rlt if np.all(e==u)])==0:
            rlt.append(u)
        count +=1
    return rlt

def all_words_in_files(data_dir="data/Syllogism/", ofile="words.txt"):
    filenames = [
        "Modus_Barbara.txt",  # 1 : 500
        "Modus_Barbari.txt",  # 2 : 500
        "Modus_Celarent_Cesare.txt",  # 3,4 : 500
        "Modus_Calemes_Camestres.txt",  # 5,6 : 500
        "Modus_Darii_Datisi.txt",  # 7,8 : 518
        "Modus_Darapti.txt",  # 9 : 500
        "Modus_Disamis_Dimatis.txt",  # 10,11 : 500
        "Modus_Baroco.txt",  # 12 : 500
        "Modus_Cesaro_Celaront.txt",  # 13,14 : 500
        "Modus_Camestros_Calemos.txt",  # 15,16 : 500
        "Modus_Bocardo.txt",  # 17 : 500
        "Modus_Bamalip.txt",  # 18 : 500
        "Modus_Ferio_Festino_Ferison_Fresison.txt",  # 19,20,21,22 : 500
        "Modus_Felapton_Fesapo.txt",  # 23,24 : 500
        ]
    words = []
    for nfile in filenames:
        with open(data_dir+nfile, "r", encoding="utf-8") as f:
            for ln in f:
                "all relation.n.01 abstraction.n.06, all possession.n.02 relation.n.01:"
                premises, _ = ln.split(":")
                wlst = [ws.split(".")[0] for ws in premises.split() if '.' in ws]
                wlst = list(set(wlst))
                nlst = [word for word in wlst if word not in words]
                words += nlst
    words.sort()
    with open(data_dir + ofile, "a+") as ofh:
        ofh.write("\n".join(words)+"\n")


def create_TB_0(ent_num, s_list):
    tb = np.array([["" for _ in range(ent_num)] for _ in range(ent_num)])
    new_s_list = []
    for (e1, e2, rel, test) in s_list:
        if e1 < e2:
            tb[e1][e2] = rel
            new_s_list.append((e1, e2, rel, test))
        elif e1 > e2:
            if rel in ["O", "D", "-D"]:
                tb[e2][e1] = rel
                new_s_list.append((e2, e1, rel, test))
            elif rel == "Pbar":
                tb[e2][e1] = "P"
                new_s_list.append((e2, e1, "P", ""))
            elif rel == "-Pbar":
                tb[e2][e1] = "-P"
                new_s_list.append((e2, e1, "-P", ""))
            elif rel == "P":
                tb[e2][e1] = "Pbar"
                new_s_list.append((e2, e1, "Pbar", ""))
            elif rel == "-P":
                tb[e2][e1] = "-Pbar"
                new_s_list.append((e2, e1, "-Pbar", ""))
            else:
                raise NotImplementedError("??? " + rel)
    return tb, new_s_list


def create_syllogism4pretrained_data(ifile="data/Syllogism/syllogism4pretrained_vec.txt",
                                     ofile="data/Syllogism/syllogism4pretrained.txt"):
    nlst = []
    with open(ifile, 'r') as ifh:
        for ln in ifh:
            "all space.n.01 attribute.n.02, all attribute.n.02 entity.n.01: all space.n.01 entity.n.01;"
            premises, conclusions = ln.split(":")
            terms = conclusions.split(";")[0].strip().split()
            s, p = terms[1], terms[2]
            nlst.append("{}: all {} {}".format(premises, s, p))
            nlst.append("{}: some-not {} {}".format(premises, s, p))
            nlst.append("{}: no {} {}".format(premises, s, p))
            nlst.append("{}: some {} {}".format(premises, s, p))
    with open(ofile, 'a+') as ofh:
        ofh.write("\n".join(nlst)+"\n")

def distance(o1, o2):
    return np.linalg.norm(np.array(o1) - np.array(o2))

def collinear(o1, o2, o3):
    d1, d2, d3 = distance(o1, o2), distance(o3, o2), distance(o1, o3)
    m = max(d1, d2, d3)
    return 2*m - d1 - d2 -d3

def rel_inv(rel):
    # if rel is a key in the dic, return dic[rel], otherwise return rel
    dic = {"P":"Pbar", "Pbar":"P", "-P": "-Pbar", "-Pbar":"-P", "D":"D", "-D":"-D"}
    return dic.get(rel, rel)


def spatialise_relation(rel):
    # if rel is a key in the dic, return dic[rel], otherwise return rel
    dic = {"all":"P", "no":"D", "some": "-D", "some-not":"-P"}
    return dic.get(rel, rel)


def consistent_rel(rel, ssyl_rel):
    # if rel is a key in the dic, return dic[rel], otherwise return rel
    dic = {"P": ["PP", "P", "EQ"],
           "Pbar": ["PPbar", "Pbar", "EQ"],
           "D": ["D"],
           "-D": ["PP", "P", "EQ", "Pbar", "PPbar", "PO","O"],
           "-P": ["PPbar", "PO", "O", "D"],
           "-Pbar": ["PP", "PO", "O", "D"]}
    if rel in dic.get(ssyl_rel, []):
        return True
    else:
        return False


def inverse_relation(rel):
    rel_dic = {'D':'D','-D':'-D','P':'Pbar', 'Pbar':'P', 'PP':'PPbar', 'PPbar':'PP', '-P':'-Pbar','-Pbar':'-P', 'PO':'PO'}
    return rel_dic.get(rel,'')


def negate_syllogistic_relation(rel):
    rel_dic = {'all':'some-not','no':'some','some':'no', 'some-not':'all'}
    return rel_dic.get(rel,'')


def rel_neg(rel):
    # if rel is a key in the dic, return dic[rel], otherwise return rel
    dic = {"P":"-P", "PP":"-PP", "Pbar":"-Pbar", "PPbar":"-PPbar", "-P": "P",
           "-Pbar":"Pbar", "-PPbar":"PPbar", "-PP": "PP", "D":"-D", "-D":"D", 'PO':'PO'}
    return dic.get(rel, rel)


def make_sentence(s, lower = False):
    if lower: s = s.lower()
    wlst = s.split()
    if wlst[0] in ['all', 'no', 'some']:
        return '{} {} are {}'.format(wlst[0], wlst[1], wlst[2])
    else:
        return 'some {} are not {}'.format(wlst[1], wlst[2])

def negate_conclusion(statement_set):
    premises = statement_set[:-1]
    conclusion = tuple([rel_neg(ele) for ele in statement_set[-1]])
    return premises + [conclusion]


def get_uniform_locations_on_sphere(dim=2, centra = 0, r=1, startLoc=[], num = 10):
    rlt, count = [], 0
    if len(startLoc)>0:
        rlt.append(2 * centra - startLoc)
    while len(rlt) < num:
        u = np.random.uniform(-1, 1, dim)
        norm = np.linalg.norm(u)
        u = u*r/norm
        u += centra
        norm2 = np.linalg.norm(u - centra)
        if r == norm2 and len([e for e in rlt if np.all(e==u)])==0:
            rlt.append(u)
        u2 = 2*centra - u
        norm2 = np.linalg.norm(u2 - centra)
        if r == norm2 and len([e for e in rlt if np.all(e == u2)]) == 0:
            rlt.append(u2)
        if r == 0: break
    return rlt


def embed_dict(data_file="data/glove.6B.50d.txt", norm=False):
    W2Es = dict()
    with open(data_file, "r", encoding="utf-8") as f:
        for ln in f.readlines():
            wlst = ln[:-1].split()
            fv = [float(n.rstrip()) for n in wlst[1:]]
            if norm:
                l = float(np.linalg.norm(fv))
                fv = [e/l for e in fv]
            W2Es[wlst[0]] = fv
    return len(wlst)-1, W2Es


def get_all_word_stems(ifile="data/Syllogism/syllogism4pretrained.txt", ofile="data/words.txt"):
    words = []
    with open(ifile, "r", encoding="utf-8") as wf:
        "some woman.n.01 reservist.n.01, all reservist.n.01 soldier.n.01: no woman.n.01 soldier.n.01"
        for ln in wf:
            wlst =[wstem.split('.')[0] for wstem in list(set([ele for ele in ln.split() if '.' in ele]))]
            words += [wd for wd in wlst if wd not in words]
    words = list(set(words))
    with open(ofile, 'a+') as ofh:
        ofh.write("\n".join(words)+"\n")


def create_bert_vec(ifile="data/glove.6B.50d.txt", ofile="data/wstem_bert_embeddings.txt", word_file="data/words.txt"):
    with open(word_file, "r", encoding="utf-8") as wf:
        word_list = [ele[:-1] for ele in wf.readlines()]
    client = BertClient()
    fh = open(ofile, 'a+')
    with open(ifile, "r", encoding="utf-8") as f:
        for ln in f:
            w = ln.split()[0]
            if w not in word_list:
                continue
            lst = client.encode([[w]], is_tokenized=True)
            vector = [str(e) for e in lst[0]]
            nln = ' '.join([w]+vector+['\n'])
            fh.write(nln)
    fh.close()


def filter_glove_list(ifile="data/glove.6B.50d.txt", ofile="data/wstem_glove.6B.50d.txt", word_file="data/words.txt"):
    with open(word_file, "r", encoding="utf-8") as wf:
        word_list = [ele[:-1] for ele in wf.readlines()]
    fh = open(ofile, 'a+')
    with open(ifile, "r", encoding="utf-8") as f:
        for ln in f:
            w = ln.split()[0]
            if w not in word_list:
                continue
            fh.write(ln)
    fh.close()


def create_TB_0(ent_num, s_list):
    tb = [["" for _ in range(ent_num)] for _ in range(ent_num)]
    new_s_list = []
    for (e1, e2, rel, test) in s_list:
        if e1 < e2:
            tb[e1][e2] = rel
            new_s_list.append((e1, e2, rel, test))
        elif e1 > e2:
            if rel in ["O", "D", "-D", "PO"]:
                tb[e2][e1] = rel
                new_s_list.append((e2, e1, rel, test))
            elif rel == "Pbar":
                tb[e2][e1] = "P"
                new_s_list.append((e2, e1, "P", ""))
            elif rel == "-Pbar":
                tb[e2][e1] = "-P"
                new_s_list.append((e2, e1, "-P", ""))
            elif rel == "P":
                tb[e2][e1] = "Pbar"
                new_s_list.append((e2, e1, "Pbar", ""))
            elif rel == "-P":
                tb[e2][e1] = "-Pbar"
                new_s_list.append((e2, e1, "-Pbar", ""))
            else:
                raise NotImplementedError("??? " + rel)
    return tb, new_s_list


# visualizing accuracy with number of epoches

def visualizaing_accuracy_with_num_of_epoches():
    filenames = ["Modus_Barbara.txt",  # 1 : 500
                 "Modus_Barbari.txt",  # 2 : 500
                 "Modus_Celarent_Cesare.txt",  # 3,4 : 500
                 "Modus_Calemes_Camestres.txt",    # 5,6 : 500
          "Modus_Darii_Datisi.txt", # 7,8 : 518
          "Modus_Darapti.txt",  # 9 : 500
          "Modus_Disamis_Dimatis.txt",  # 10,11 : 500
          "Modus_Baroco.txt",   # 12 : 500
          "Modus_Cesaro_Celaront.txt",  # 13,14 : 500
          "Modus_Camestros_Calemos.txt",    # 15,16 : 500
          "Modus_Bocardo.txt",  # 17 : 500
          "Modus_Bamalip.txt",  # 18 : 500
          "Modus_Ferio_Festino_Ferison_Fresison.txt",   # 19,20,21,22 : 500
          "Modus_Felapton_Fesapo.txt",  # 23,24 : 500
            ]
    epoches = [10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90]
    fpath = '/Users/tdong/pvgit/vnn/vnn-code'
    fdict = dict()
    for fn in filenames:
        fdict[fn]=[]
    for ep in epoches:
        fname = os.path.join(fpath, "Syllogism_{}ep.tbl.txt".format(ep))
        if os.path.isfile(fname):
            with open(fname, 'r') as ifh:
                for ln in ifh:
                    words=ln[:-1].split()
                    for fn in filenames:
                        if words[0] == fn:
                            if len(fdict[fn])>0:
                                if fdict[fn][-1] != 1.0:
                                    fdict[fn].append(float(words[-1]))
                            else:
                                fdict[fn].append(float(words[-1]))

    colors = [[0., 0., 1],[0.0, 0.7, 0.2],[1, 0., .0], [0.4, 1, 0],[0.5, 0., 1.0],[.5, 0.6, .0],
              [0., 0.5, 0.6], [0.3, 0.9, 0.3], [1, 0.2, 1.0], [0.2, 0.8, 0.7], [0.9, 0.6, .3], [0.2, 0.8, 1.0],
              [0.1, .1, 0.1], [0.1, .1, .8], [0.8, .1, .2],  [0.4, 1.0, 0.4], [0.8, 0, 1.0], [0, 1.0, 1.0]
               ]
    markers = ['o', 'v', '^', '8', 's', 'p', 'P', '*', 'X', 'd', 'D', '.', '<', '>']
    i = 0
    fig = plt.figure(figsize=(10, 6))
    for fn in filenames:
        tag = fn[6:-4]
        plt.plot(epoches[:len(fdict[fn])], fdict[fn], color=colors[i],marker=markers[i], label=tag)
        i += 1
    plt.legend()
    plt.xlabel('Number of training epochs')
    plt.ylabel('Accuracy')
    plt.show()
    # plt.savefig('syllogism_pic.png', dpi=fig.dpi)


def visualizaing_accuracy_of_family_data(ifile="data/family_result.txt"):
    entities=[]
    accuracies=[]
    precisions = []
    recalls = []
    F1 = []
    numOfFamilies=[]
    numOfT = []
    numOfF = []
    with open(ifile, 'r') as ifh:
        for ln in ifh:
            ln = ln.replace(',','.')
            wlst = ln.split()
            if int(wlst[1])>5:
                entities.append(int(wlst[0]))
                accuracies.append(round(float(wlst[7]), 3))
                precisions.append(round(float(wlst[8]), 3))
                recalls.append(round(float(wlst[9]), 3))
                F1.append(round(float(wlst[10]), 3))
                numOfFamilies.append(wlst[1])
                numOfT.append(wlst[2])
                numOfF.append(wlst[3])
    fig = plt.figure(figsize=(10, 6))
    plt.ylim(0.7,1.0)
    #plt.plot(entities, accuracies, marker='^', color=[1,0,0], label=" FLY: Number of Families\n Prp: Number of Propositions")
    plt.plot(entities, accuracies, marker='^', color=[1, 0, 0], label=" Accuracy")
    plt.plot(entities, precisions, marker='o', color=[0, 1, 0], label=" Precision")
    plt.plot(entities, recalls, marker='8', color=[0, 0, 1], label=" Recall")
    plt.plot(entities, F1, marker='8', color=[0, 0.6, 0.6], label=" F1")
    plt.legend()

    for i,j in enumerate(range(len(entities))):
        #plt.annotate("\n".join(["FLY:"+numOfFamilies[i], "Prp:"+str(int(numOfT[i]) + int(numOfF[i]))]), (entities[i], accuracies[i]))
        plt.annotate(str(accuracies[i]), (entities[i], accuracies[i]))
        plt.annotate(str(precisions[i]), (entities[i], precisions[i]))
        plt.annotate(str(recalls[i]), (entities[i], recalls[i]))
        plt.annotate(str(F1[i]), (entities[i], F1[i]))

    plt.xlabel('Number of Family Members')
    plt.xticks(entities, entities)
    # plt.show()
    plt.savefig('family_pic_3A.png', dpi=fig.dpi)


def create_table_for_family_data(ifile="data/submitted_family_data_statistics.txt"):
    with open(ifile, 'r') as ifh:
        ln0 = next(ifh)
        ln0 = ln0.replace('#', '\#')
        wlst = ln0[:-1].split()
        wdict = defaultdict(list)
        for ln in ifh:
            nlst = ln[:-1].split()
            for i in range(len(nlst)):
                if int(nlst[1])>5:
                    wdict[wlst[i]].append(nlst[i])
    columnStr = "l|" + "c"*len(wdict[wlst[0]])
    lns = []
    for w in wlst:
        lns.append('&'.join([w]+wdict[w]))
    strTable ="\\beginQtableZ\captionQ{C}Z\labelQ{L}Z\centering\hspace*Q-1.2emZ\scaleboxQ0.9ZQ" \
              "\\beginQtabularZQ{N}Z" \
              "\\toprule" \
              "{S}" \
              "\\\\\\bottomrule\endQtabularZZ\endQtableZ".format(C='Datasets extracted from DBpedia for reasoning with family relations',
                                                                 N=columnStr, L='fam_data', S='\\\\'.join(lns))
    strTable = strTable.replace('Q', '{')
    strTable = strTable.replace('Z', '}')
    print(strTable)


def create_error_records(ifile0="data/family_test_detail.txt", ifile1="data/Family/cFamily_relation_test.txt"):
    eDict = dict()
    with open(ifile0, 'r') as ifh0:
        for ln in ifh0:
            ln = ln.replace(",", ".")
            wlst = ln[:-1].split()
            id, acc = int(wlst[0]), float(wlst[5])
            if acc != 1:
                eDict[id] = acc
    eLst = [id*4 for id in eDict.keys()]
    eLst.sort()
    with open(ifile1, 'r') as ifh1:
        i = 0
        eRecords = []
        for eId in eLst:
            while i != eId:
                i += 1
                print(i)
                print(ifh1.readline())
            fln = ifh1.readline()
            if not fln.startswith("*WRONG"):
                eRecords.append("".join([fln, ifh1.readline(),ifh1.readline(), ifh1.readline()]))
            i += 4
    with open("data/Family/eFamily_relation_test.txt", 'w') as ofh:
        ofh.writelines(''.join(eRecords))


def dict2matrix(rstat):
    X, Y, Z = [], [], []
    minT, maxT = (0,0, 1000000000), (0,0,0)
    for R in [1, 2, 3, 4, 5, 20, 30, 40, 50]:
        xlst, ylst, zlst = [], [], []
        for d in [2, 10, 50, 100, 200, 300, 400,  500, 600, 700, 800]:
            xlst.append(R)
            ylst.append(d)
            zlst.append(rstat[d][R])
            if rstat[d][R] > 0 and rstat[d][R] < minT[2]:
                minT = (R, d, rstat[d][R])
            if rstat[d][R] > 0 and rstat[d][R] > maxT[2]:
                maxT = (R, d, rstat[d][R])
        X.append(np.array(deepcopy(xlst)))
        Y.append(np.array(deepcopy(ylst)))
        Z.append(np.array(deepcopy(zlst)))
    return np.array(X), np.array(Y), np.array(Z), minT, maxT


def view_dim_initR_time_3d(ifile0="data/partial_stat_Jan02.txt"):
    rstat = defaultdict(defaultdict)
    pstat = defaultdict(defaultdict)
    for R in [1, 2, 3, 4, 5, 20, 30, 40, 50]:
        for d in [2, 10, 50, 100, 200, 300, 400,  500, 600, 700,  800]:
            rstat[R][d] = -1000
    for R in [1, 2, 3, 4, 5, 20, 30, 40, 50]:
        for d in [50, 768]:
            pstat[R][d] = -1000

    with open(ifile0, 'r') as fh:
        for ln in fh:
            wlst = ln.split()
            T, R = float(wlst[2]), int(wlst[5])
            if 'random' in ln:
                D = int(wlst[0].replace('R', ' ').replace('random', ' ').split()[1])
                rstat[R][D]=T
            elif 'pretrain' in ln:
                D = int(wlst[0].replace('R', ' ').replace('pretrain', ' ').split()[1])
                if D == 1:
                    pstat[R][50] = T
                elif D == 2:
                    pstat[R][768] = T

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([0, 60])
    # X, Y, Z = axes3d.get_test_data(0.05)
    # X, Y, Z, minT, maxT = dict2matrix(rstat)

    X, Y, Z = [], [], []
    minT, maxT = (0, 0, 1000000000), (0, 0, 0)
    for R in [1, 2, 3, 4, 5, 20, 30, 40, 50]:
        xlst, ylst, zlst = [], [], []
        for d in [2, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800]:
            xlst.append(R)
            ylst.append(d)
            zlst.append(rstat[R][d])
            ax.scatter(R, d, rstat[R][d], c='b', marker='+')
            if rstat[R][d] > 0 and rstat[R][d] < minT[2]:
                minT = (R, d, rstat[R][d])
            if rstat[R][d] > 0 and rstat[R][d] > maxT[2]:
                maxT = (R, d, rstat[R][d])
        if R < 3:
            ax.plot_wireframe(np.array([np.array(deepcopy(xlst))]),
                          np.array([np.array(deepcopy(ylst))]),
                          np.array([np.array(deepcopy(zlst))]),
                          color=[ 10*R/50, 0, 0],
                          rstride=10, cstride=10)
        elif R < 30:
            ax.plot_wireframe(np.array([np.array(deepcopy(xlst))]),
                              np.array([np.array(deepcopy(ylst))]),
                              np.array([np.array(deepcopy(zlst))]),
                              color=[0, R / 50, 0],
                              rstride=10, cstride=10)
        else:
            ax.plot_wireframe(np.array([np.array(deepcopy(xlst))]),
                              np.array([np.array(deepcopy(ylst))]),
                              np.array([np.array(deepcopy(zlst))]),
                              color= [R / 50, 0, 0],
                              rstride=10, cstride=10)
        X.append(np.array(deepcopy(xlst)))
        Y.append(np.array(deepcopy(ylst)))
        Z.append(np.array(deepcopy(zlst)))

    print("X", X)
    print("Y", Y)
    print("Z", Z)
    X, Y, Z = np.array(X), np.array(Y), np.array(Z)
#    cset = ax.contour(X, Y, Z, cmap=cm.coolwarm)

#    ax.clabel(cset, fontsize=9, inline=1)

    ax.scatter(minT[0], minT[1], minT[2], c='r', marker='o')
    ax.scatter(maxT[0], maxT[1], maxT[2], c='b', marker='^')
    print('minT', minT[0], minT[1], minT[2])
    print('maxT', maxT)

    ax.set_xlabel('radius of the initial configuration')
    ax.set_ylabel('dimension of Euler diagram')
    ax.set_zlabel('time for 7000 tasks')
    plt.show()

def view_dim_initR_time_2d(ifile0="data/partial_stat_Jan02.txt"):
    rstat = defaultdict(defaultdict)
    pstat = defaultdict(defaultdict)
    for R in [1, 2, 3, 4, 5, 20, 30, 40, 50]:
        for d in [2, 10, 50, 100, 200, 300, 400,  500, 600, 700,  800]:
            rstat[R][d] = 0
    for R in [1, 2, 3, 4, 5, 20, 30, 40, 50]:
        for d in [50, 768]:
            pstat[R][d] = 0

    with open(ifile0, 'r') as fh:
        for ln in fh:
            wlst = ln.split()
            T, R = float(wlst[2])/7000, int(wlst[5])
            if 'random' in ln:
                D = int(wlst[0].replace('R', ' ').replace('random', ' ').split()[1])
                rstat[R][D]=T
            elif 'pretrain' in ln:
                D = int(wlst[0].replace('R', ' ').replace('pretrain', ' ').split()[1])
                if D == 1:
                    pstat[R][50] = T
                elif D == 2:
                    pstat[R][768] = T

    X, Y, Z = [], [], []
    minT, maxT = (0, 0, 1000000000), (0, 0, 0)
    clst=['0.15','g','y','b','k','m', 'c', 'r', '#005522', '#035e22', '#0e0e0e', '#035e22', '#0e0e0e']
    mklst = ['o', 'v', '+', 'X', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
    loc = []
    dlst = []
    i = 0
    rd50 = ['random50']
    rd768 = ['random768']
    for R in [1, 2, 3, 4, 5, 20, 30, 40, 50]:
        xlst, ylst, zlst = [], [], []
        flag = False
        for d in [2, 10, 50, 100, 200, 300, 400, 500, 600, 700, 768, 800]:
            xlst.append(R)
            ylst.append(d)
            zlst.append(rstat[R][d])
            if d == 50:  rd50.append("{:.2f}".format(rstat[R][d]))
            if d == 768:  rd768.append("{:.2f}".format(rstat[R][d]))
            if d not in dlst and not flag:
                dlst.append(d)
                if R == 30:
                    loc.append([d + 7, rstat[R][d] - 0.15])
                elif R in [3, 40]:
                    loc.append([d - 20, rstat[R][d] + 0.05])
                else:
                    loc.append([d + 7, rstat[R][d]+0.05])
                flag = True
            if rstat[R][d] > 0 and rstat[R][d] < minT[2]:
                minT = (R, d, rstat[R][d])
            if rstat[R][d] > 0 and rstat[R][d] > maxT[2]:
                maxT = (R, d, rstat[R][d])
        plt.subplot(1, 1, 1)
        plt.ylim([1, 4])
        plt.scatter(np.array(deepcopy(ylst)), np.array(deepcopy(zlst)), c=clst[i], marker=mklst[i])
        plt.plot(np.array(deepcopy(ylst)), np.array(deepcopy(zlst)), clst[i])
        plt.text(loc[i][0], loc[i][1], 'R={}'.format(R), color=clst[i])
        plt.xlabel('dimension of Euler diagram')
        plt.ylabel('time for one tasks (in seconds)')
        i += 1
    i = 0
    glv = ['GloVE50']
    bert = ['BERT768']
    for R in [1, 2, 3, 4, 5, 20, 30, 40, 50]:
        xlst, ylst, zlst = [], [], []
        flag = False
        for d in [50, 768]:
            xlst.append(R)
            ylst.append(d)
            zlst.append(pstat[R][d])
            if d == 50:  glv.append("{:.2f}".format(pstat[R][d]))
            if d == 768:  bert.append("{:.2f}".format(pstat[R][d]))
            if d not in dlst and not flag:
                dlst.append(d)
                if R == 30:
                    loc.append([d + 7, pstat[R][d] - 0.15])
                elif R in [3, 40]:
                    loc.append([d - 20, pstat[R][d] + 0.05])
                else:
                    loc.append([d + 7, pstat[R][d] + 0.05])
                flag = True

        plt.subplot(1, 1, 1)
        plt.ylim([1, 4])
        plt.scatter(np.array(deepcopy(ylst)), np.array(deepcopy(zlst)), c=clst[i], marker=mklst[i])
        # plt.text(loc[i][0], loc[i][1], 'R={}'.format(R), color=clst[i])
        i += 1
        # ax.grid()\
    plt.show()
    print('&'.join(rd50))
    print('&'.join(glv))
    print('&'.join(rd768))
    print('&'.join(bert))


def info_dim_initR_time_2d(ifile0="data/partial_stat_Jan02.txt"):
    rstat = defaultdict(defaultdict)
    pstat = defaultdict(defaultdict)
    for R in [1, 2, 3, 4, 5, 20, 30, 40, 50]:
        for d in [2, 10, 50, 100, 200, 300, 400,  500, 600, 700,  800]:
            rstat[R][d] = 0
    for R in [1, 2, 3, 4, 5, 20, 30, 40, 50]:
        for d in [50, 768]:
            pstat[R][d] = 0

    with open(ifile0, 'r') as fh:
        for ln in fh:
            wlst = ln.split()
            T, R = float(wlst[2])/7000, int(wlst[5])
            if 'random' in ln:
                D = int(wlst[0].replace('R', ' ').replace('random', ' ').split()[1])
                rstat[R][D]=T
            elif 'pretrain' in ln:
                D = int(wlst[0].replace('R', ' ').replace('pretrain', ' ').split()[1])
                if D == 1:
                    pstat[R][50] = T
                elif D == 2:
                    pstat[R][768] = T

    X, Y, Z = [], [], []
    minT, maxT = (0, 0, 1000000000), (0, 0, 0)
    clst=['0.15','g','y','b','k','m', 'c', 'r', '#005522', '#035e22', '#0e0e0e', '#035e22', '#0e0e0e']
    mklst = ['o', 'v', '+', 'X', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
    loc = []
    dlst = []
    i = 0
    rd50 = ['random50']
    rd768 = ['random768']
    for R in [1, 2, 3, 4, 5, 20, 30, 40, 50]:
        xlst, ylst, zlst = [], [], []
        flag = False
        for d in [2, 10, 50, 100, 200, 300, 400, 500, 600, 700, 768, 800]:
            xlst.append(R)
            ylst.append(d)
            zlst.append(rstat[R][d])
            if d == 50:  rd50.append("{:.2f}".format(rstat[R][d]))
            if d == 768:  rd768.append("{:.2f}".format(rstat[R][d]))
            if d not in dlst and not flag:
                dlst.append(d)
                if R == 30:
                    loc.append([d + 7, rstat[R][d] - 0.15])
                elif R in [3, 40]:
                    loc.append([d - 20, rstat[R][d] + 0.05])
                else:
                    loc.append([d + 7, rstat[R][d]+0.05])
                flag = True
            if rstat[R][d] > 0 and rstat[R][d] < minT[2]:
                minT = (R, d, rstat[R][d])
            if rstat[R][d] > 0 and rstat[R][d] > maxT[2]:
                maxT = (R, d, rstat[R][d])
        plt.subplot(1, 1, 1)
        plt.ylim([1, 4])
        plt.scatter(np.array(deepcopy(ylst)), np.array(deepcopy(zlst)), c=clst[i], marker=mklst[i])
        plt.plot(np.array(deepcopy(ylst)), np.array(deepcopy(zlst)), clst[i])
        plt.text(loc[i][0], loc[i][1], 'R={}'.format(R), color=clst[i])
        plt.xlabel('dimension of Euler diagram')
        plt.ylabel('time for one tasks (in seconds)')
        i += 1
    i = 0
    glv = ['GloVE50']
    bert = ['BERT768']
    for R in [1, 2, 3, 4, 5, 20, 30, 40, 50]:
        xlst, ylst, zlst = [], [], []
        flag = False
        for d in [50, 768]:
            xlst.append(R)
            ylst.append(d)
            zlst.append(pstat[R][d])
            if d == 50:  glv.append("{:.2f}".format(pstat[R][d]))
            if d == 768:  bert.append("{:.2f}".format(pstat[R][d]))
            if d not in dlst and not flag:
                dlst.append(d)
                if R == 30:
                    loc.append([d + 7, pstat[R][d] - 0.15])
                elif R in [3, 40]:
                    loc.append([d - 20, pstat[R][d] + 0.05])
                else:
                    loc.append([d + 7, pstat[R][d] + 0.05])
                flag = True

        plt.subplot(1, 1, 1)
        plt.ylim([1, 4])
        plt.scatter(np.array(deepcopy(ylst)), np.array(deepcopy(zlst)), c=clst[i], marker=mklst[i])
        # plt.text(loc[i][0], loc[i][1], 'R={}'.format(R), color=clst[i])
        i += 1
        # ax.grid()\
    plt.show()
    print('&'.join(rd50))
    print('&'.join(glv))
    print('&'.join(rd768))
    print('&'.join(bert))


def get_sat_list_from_dir(root_dir="", dir_name=""):
    true_list = []
    for fname in os.listdir(root_dir + dir_name):
        if "log_data_" in fname:
            # print(fname)
            with open(root_dir + dir_name+fname, 'r') as ifh:
                for ln in ifh:
                    if  ln.startswith(" predict True"):
                        true_list.append(fname)
                        break
    print(dir_name, len(true_list))
    return true_list


def get_wrong_unsat_list(dir_root="/home/user/dongt/git/ENN22S/", dir_name="_runs/...", sat_file='data/Syllogism/sat_list.txt'):
    with open(dir_root + sat_file, 'r') as ifh:
        slist = [ele[:-1] for ele in ifh.readlines()]
    true_list = get_sat_list_from_dir(root_dir=dir_root, dir_name=dir_name)
    wrong_unsat_list = [ele for ele in slist if ele not in true_list]
    return wrong_unsat_list


def ex5_get_sat_list(dir_root="/home/user/dongt/git/ENN22S/_runs/", ofile='sat_list.txt'):
    sat_list = []

    dir_pat_pt = "Syllogism_RS0_WE_fc{}pt{}cop{}/"
    for fc in [0,1,2]:
        for pt in [1,2]:
            for cop in [0,1]:
                dname = dir_pat_pt.format(fc,pt, cop)
                list0 = get_sat_list_from_dir(root_dir=dir_root, dir_name=dname)
                print(dname, list0)
                sat_list += [ele for ele in list0 if ele not in sat_list]

    #dir_pat_rd = "Syllogism_RS0_random{}R{}cop0/"
    #for R in [1, 2, 3, 4, 5, 20, 30]:
    #    for d in [50, 768]:
    #        list0 = get_sat_list_from_dir(root_dir=dir_root, dir_name=dir_pat_rd.format(d, R))
    #        sat_list += [ele for ele in list0 if ele not in sat_list]

    with open(ofile, 'a+') as ofh:
        sat_list.sort()
        ofh.write("\n".join(sat_list)+"\n")


def ex5_wrong_unsat_list(dir_root="/home/user/dongt/git/ENN22S/", dir_pat = "_runs/Syllogism_RS0_WE_fc{}pt{}cop{}/",
                         sat_file='data/Syllogisms/sat_list.txt'):

    for fc in [0,1,2]:
        for pt in [1,2]:
            for cop in [0]:
                dname = dir_pat.format(fc,pt, cop)
                list0 = get_wrong_unsat_list(dir_root=dir_root, dir_name= dname, sat_file=sat_file)
                print(dname, list0[:10])


def ex5_refute_value_of_pre_trained_vector(sat_list_file="",
                                           dir_root="/home/user/dongt/git/ENN22S/runs_rlt/",
                                           ofile='wrong_unsat_prediction.txt'):
    rlt_dict = defaultdict(dict)
    with open(sat_list_file, 'r') as ifh:
        sat_list = [ele[:-1] for ele in ifh.readlines()]

    dir_pat_rd = "Syllogism_RS0_random{}R{}cop0/"
    dir_pat_fc = "Syllogism_RS0_WE_fc{}pt{}cop{}/"
    for R in [1, 2, 3, 4, 5, 20, 30, 40]:
        for d in [50, 768]:
            dname = dir_pat_rd.format(d, R)
            list0 = get_sat_list_from_dir(root_dir=dir_root, dir_name=dname)
            wrong_unsat_list = [ele for ele in sat_list if ele not in list0]
            rlt_dict[dname] = wrong_unsat_list

    for fc in [0,1,2]:
        for pt in [1, 2]:
            for cop in [0, 1]:
                dname = dir_pat_fc.format(fc, pt, cop)
                list0 = get_sat_list_from_dir(root_dir=dir_root, dir_name=dname)
                wrong_unsat_list = [ele for ele in sat_list if ele not in list0]
                rlt_dict[dname] = wrong_unsat_list

    lns = []
    for k, v in rlt_dict.items():
        lns.append(" ".join([k, str(len(v))] + v))
        if len(v) == 0:
            print(k)

    with open(ofile, 'a+') as ofh:
        ofh.write('\n'.join(lns))


def ex5_check_any_false(dir_root="/home/user/dongt/git/ENN22S/_runs/", fix_center=False):
    results = defaultdict(dict)
    for PreTrain in ['Pretrain', 'random']:
        if PreTrain.startswith('Pretrain'):
            wlst = [1, 2]
            dir_pattern = "Syllogism_RS_WE_fc{}pt{}R{}cop{}"
        else:
            wlst = [50, 768]
            dir_pattern = "Syllogism_RS_random{}R{}cop{}"

        for w in wlst:
            for DIM in [1, 2, 3, 4, 5, 20, 30, 40]:
                for cop in [0, 1]:
                    if PreTrain.startswith('Pretrain') and not fix_center:
                        dname = dir_pattern.format('0', str(w), str(DIM), str(cop))
                    elif PreTrain.startswith('Pretrain') and fix_center:
                        dname = dir_pattern.format('1', str(w), str(DIM), str(cop))
                    else:
                        dname = dir_pattern.format(str(w), str(DIM), str(cop))

                    true_num = 0
                    for fname in os.listdir(dir_root+dname):
                        if "log_data_" in fname:
                            with open(dir_root+dname+'/'+fname, 'r') as f:
                                for ln in f:
                                    if 'predict' in ln:
                                        if ln.split()[1] == "True":
                                            true_num += 1
                                        else:
                                            print(ln)
                    results[dname] = true_num
    pprint(results)


def ex5_error_patterns_when_fixing_center(dir_root="/home/user/dongt/git/ENN22S/_runs/"):
    results = defaultdict(dict)
    wlst = [1, 2]
    dir_pattern = "Syllogism_RS_WE_fc1pt{}R{}cop{}"

    for w in wlst:
        for DIM in [1, 2, 3, 4, 5, 20, 30, 40]:
            for cop in [0, 1]:
                dname = dir_pattern.format(str(w), str(DIM), str(cop))
                flst = []
                for fname in os.listdir(dir_root+dname):
                    if "log_data_" in fname:
                        with open(dir_root+dname+'/'+fname, 'r') as f:
                            for ln in f:
                                if 'predict' in ln:
                                    if ln.split()[1] == "False":
                                        start = ln.index("raw_input")
                                        flst.append(ln[start:])
                results[dname] = flst
    pprint(results)


def ex5_find_total_runtime(dir_root="/home/user/dongt/git/ENN22S/_runs/", fix_center=False):
    results = defaultdict(dict)
    for PreTrain in ['Pretrain', 'random']:
        if PreTrain.startswith('Pretrain'):
            wlst = [1, 2]
            dir_pattern = "Syllogism_RS_WE_fc{}pt{}R{}cop{}"
        else:
            wlst = [50, 768]
            dir_pattern = "Syllogism_RS_random{}R{}cop{}"

        for w in wlst:
            for DIM in [1, 2, 3, 4, 5, 20, 30, 40]:
                for cop in [0, 1]:
                    if PreTrain.startswith('Pretrain') and not fix_center:
                        dname = dir_pattern.format('0', str(w), str(DIM), str(cop))
                    elif PreTrain.startswith('Pretrain') and fix_center:
                        dname = dir_pattern.format('1', str(w), str(DIM), str(cop))
                    else:
                        dname = dir_pattern.format(str(w), str(DIM), str(cop))

                        for fname in os.listdir(dir_root+dname):
                            if "log_data_" in fname:
                                with open(dir_root+dname+'/'+fname, 'r') as f:
                                    for ln in f:
                                        if 'ref_R' in ln:
                                            print(dname, ln)
                                            #'Syllogism_RS22_Pretrain1R2opt0'
                                            #total time: data_0.8388121128082275:  1050 True
                                            dicName, restv = dname.split('_')[-1].split('R')
                                            rvalue, optv = restv.split('opt')
                                            print(float(ln.split()[2])/2537)
                                            average_time = float(ln.split()[2])/2537
                                            results[dicName][int(rvalue)] = str(average_time)[:5]
    print(results)
    for key in results.keys():
        sln = [key]
        for r in [1, 2, 3, 4, 5,  20, 30, 40]:
            sln.append(results[key][r])
        print('&'.join(sln))



def ex5_find_fastest_case(dir_root="/home/user/dongt/git/ENN22S/_runs/"):
    """
    fixed center, non-fixed center, random which is the fastest?
    :param dir_root:
    :return:
    """
    results = defaultdict(dict)
    for PreTrain in ['Pretrain', 'random']:
        if PreTrain.startswith('Pretrain'):
            wlst = [1, 2]
            dir_pattern = "Syllogism_RS_WE_fc{}pt{}R{}cop{}"
        else:
            wlst = [50, 768]
            dir_pattern = "Syllogism_RS_random{}R{}cop{}"

        for w in wlst:
            for DIM in [1, 2, 3, 4, 5, 20, 30, 40]:
                for cop in [0, 1]:
                    fc1_name = dir_pattern.format('1', str(w), str(DIM), str(cop))
                    fc0_name = dir_pattern.format('0', str(w), str(DIM), str(cop))
                    ran_name = dir_pattern.format(str(w), str(DIM), str(cop))

                    for fname in os.listdir(dir_root+fc1_name):
                        if "log_data_" in fname:
                            with open(dir_root+fc1_name+'/'+fname, 'r') as f:
                                for ln in f:
                                    if 'ref_R' in ln:
                                        print(dname, ln)
                                        #'Syllogism_RS22_Pretrain1R2opt0'
                                        #total time: data_0.8388121128082275:  1050 True
                                        dicName, restv = dname.split('_')[-1].split('R')
                                        rvalue, optv = restv.split('opt')
                                        print(float(ln.split()[2])/2537)
                                        average_time = float(ln.split()[2])/2537
                                        results[dicName][int(rvalue)] = str(average_time)[:5]
    print(results)
    for key in results.keys():
        sln = [key]
        for r in [1, 2, 3, 4, 5,  20, 30, 40]:
            sln.append(results[key][r])
        print('&'.join(sln))

def list_valid_results(data_dir="", dir_pattern=""):
    results = defaultdict(dict)
    valid_num = [0, 11, 133, 134, 141,  142, 15, 195, 203, 225, 229,   233, 237, 25, 3, 33, 34, 37, 38, 41, 45, 59, 63, 69]
    for DIM in [2, 3, 15, 30, 100, 200, 2000, 10000]:
        for COP in [0]:
            for RR in [10]:
                for R in [10]: #, 2, 3, 4, 5, 10, 20, 30, 40, 50]:
                    dname = dir_pattern.format(str(DIM), str(COP), str(RR), str(R))
                    print(os.getcwd())
                    valid_list = []
                    for fname in os.listdir(data_dir+dname):
                        if not fname.startswith("log_"):
                            continue
                        _, _, num = fname[:-4].split("_")
                        num = int(num)
                        with open(data_dir+dname+"/"+fname, 'r') as fh:
                            for ln in fh:
                                if 'True' in ln:
                                    valid_list.append(ln)
                                    if num not in valid_num:
                                        print(dname, fname, ln)
                    results[dname] = valid_list
                    print(dname, len(valid_list))
    #pprint(results)


def list_number_of_it(data_dir="", dir_pattern="", RR=50):
    for DIM in [2, 3, 15, 30, 100, 200, 2000, 10000]:
        for cop in [0]:
            dic = defaultdict()
            dname = dir_pattern.format(str(DIM), str(cop))
            print(os.getcwd())
            for fname in os.listdir(data_dir+dname):
                if not fname.startswith("log_"):
                    continue
                it_num = 0
                with open(data_dir+dname+"/"+fname, 'r') as fh:
                    for ln in fh:
                        if ln.startswith("epoch_0 loop_1"): it_num = 1
                        elif ln.startswith("epoch_0 loop_2"): it_num = 2
                        elif ln.startswith("epoch_0 loop_3"): it_num = 3

                        if "predict False" in ln:
                            if DIM not in dic.keys(): dic[DIM] = defaultdict()
                            if cop not in dic[DIM].keys(): dic[DIM][cop] = defaultdict()
                            if it_num not in dic[DIM][cop]:
                                dic[DIM][cop][it_num] = 1
                            else:
                                dic[DIM][cop][it_num] += 1
            pprint(dname)
            pprint(dic)


def generate_chained_syllogism(N, opath = 'data/ValidSyllogism/'):
    """
    generate long chained syllogism, with N terms.
    :param N:
    :return:
    """
    relLst = ['all', 'no', 'some', 'some-not']
    termLst = ["S"] + ["M"+str(e) for e in range(N)][1:-1]+["P"]
    pairLst = list(zip(termLst[:-1], termLst[1:]))
    premises = [", ".join(e) for e in create_syllogistic_statements(pairLst)]
    conclusions = [" ".join([e[0]] + list(e[1])) for e in itertools.product(relLst, [["S", "P"]])]
    rlt =[", ".join(e) for e in [list(e[:1]) + [e[1]]  for e in itertools.product(premises, conclusions)]]
    with open(opath+"Possible_Syllogism_"+str(N)+".txt", 'a+') as ofh:
        ofh.write('\n'.join(rlt))
    return rlt


def create_syllogistic_statements(pairLst):
    relLst = ['all', 'no', 'some', 'some-not']
    if len(pairLst) == 1:
        pair = pairLst[0]
        return [[" ".join([e[0]] + list(e[1]))] for e in itertools.product(relLst, itertools.permutations(pair))]
    else:
        return [e[0] + list(e[1]) for e in itertools.product(create_syllogistic_statements([pairLst[0]]),
                                                                create_syllogistic_statements(pairLst[1:]))]


def equal_syl(S1, S2):
    S1lst = S1.split(' ')
    S2lst = S2.split(' ')
    if S1lst[0] == S2lst[0] and S1lst[1] == S2lst[1] and S1lst[2] == S2lst[2]:
        return True
    elif S1lst[0] not in ['all', 'some-not'] and S1lst[0] == S2lst[0] and S1lst[2] == S2lst[1] and S1lst[1] == S2lst[2]:
        return True


def exchange_role(s):
    wlst = s.split()
    return ' '.join([wlst[0], wlst[2], wlst[1]])

def create_valid_syllogistic_reasoning(N, ifile_prefix="data/ValidSyllogism/valid_syllogism_"):
    ifile = "{}{}.txt".format(ifile_prefix, N - 1)
    hd_dic = defaultdict(list)
    with open(ifile, 'r') as ifh:
        for ln in ifh.read().split('\n'):
            ln = ln.replace('P', 'M{}'.format(N - 2))
            clauses = [e.strip() for e in ln.split(',')]
            if len(clauses[0]) > 0:
                hd_dic[clauses[-1]].append(', '.join(clauses[:-1]))
                if clauses[-1] == 'all S M2':
                    hd_dic['all M2 S'].append(', '.join([exchange_role(e) for e in clauses[:-1]]))
    hd_dic['some M2 S'] = hd_dic['some S M2']
    hd_dic['no M2 S'] = hd_dic['no S M2']

    rfile = "{}{}.txt".format(ifile_prefix, 3)
    tl_dic = defaultdict(list)
    with open(rfile, 'r') as ifh:
        for ln in ifh.read().split('\n'):
            ln = ln.replace('M1', 'M{}'.format(N - 2))
            clauses = [e.strip() for e in ln.split(',')]
            if len(clauses[0]) > 0:
                tl_dic[clauses[0]].append(', '.join(clauses[1:]))

    rlt = []
    for (H, T) in itertools.product(hd_dic.keys(), tl_dic.keys()):
        if H=='some M2 S' and T=='some M2 S':
            print()
        if equal_syl(H, T):
            for (A, B) in itertools.product(hd_dic[H], tl_dic[T]):
                new = '{}, {}'.format(A, B)
                if new not in rlt:
                    rlt.append(new)
    rlt = list(set(rlt))
    print(rlt)
    print(len(rlt))
    ofile = "{}{}.txt".format(ifile_prefix, N)
    with open(ofile, 'w+') as ofh:
        ofh.write('\n'.join(rlt))


def get_syllogism_not_in_valid_list(N, ifile='_runs/data_March31_MChoice/multiple_choice_s{}/log_data_{}.txt',
                                    ref_file="data/ValidSyllogism/valid_syllogism_{}.txt",
                                    ofile = "data/ValidSyllogism/incorrect_choice.txt"):
    ref_file = ref_file.format(N)
    rlt0 = []
    for i in range(120):
        ifname = ifile.format(N, 120*(N-3)+i)
        with open(ifname, 'r') as ifh:
            lines = [ln.strip().split('[')[1].strip(']') for ln in ifh.readlines() if 'predict True' in ln]
            lines = [ln.replace('\'', '') for ln in lines]
            if len(lines) > 0:
                rlt0 += lines

    with open(ref_file, 'r') as rfh:
        rlines = [ln.strip() for ln in rfh.readlines()]

    rlt = [ln for ln in rlt0 if ln not in rlines]
    if len(rlt) > 0:
        with open(ofile, 'a+') as ofh:
            ofh.write('\n')
            ofh.write('\n'.join(rlt))


def change_relations(mc):
    def update(statement):
        R, A, B = statement.split()
        return ' '.join([random.choice(relLst), A, B])
    relLst = ['all', 'no', 'some', 'some-not']
    wlst = mc.split(', ')
    new_wlst = [update(e) for e in wlst]
    return ', '.join(new_wlst)


def create_multiple_choices(N, ifile_prefix="data/ValidSyllogism/valid_syllogism_",
                            ofile_prefix="data/multiple_choices/syllogism_"):
    ifile = "{}{}.txt".format(ifile_prefix, N)
    rlt = []
    with open(ifile, 'r') as ifh:
        vlst = [ln.strip() for ln in ifh.read().split('\n') if len(ln)> 3]
        mchoices = random.sample(vlst, 24)
        for mc in mchoices:
            task = [deepcopy(mc)]
            while True:
                noise = change_relations(mc)
                if noise not in vlst and noise not in task:
                    task.append(noise)
                if len(task) == 5: break
            random.shuffle(task)
            rlt += task
    ofile = "{}{}.txt".format(ofile_prefix, N)
    with open(ofile, 'w+') as ofh:
        ofh.write('\n'.join(rlt))


def find_valid_taskId(infile="_runs/data_April05_MChoice/True_predictions.txt",
                      outfile="data/multiple_choice/valid_list.txt"):
    with open(infile, 'r') as ifh:
        rlst = [ln.split('.')[0].split('_')[-1] for ln in ifh.readlines()]
    ofh = open(outfile, 'w+')
    ofh.write('\n'.join(rlst))
    print("["+','.join(rlst)+"]")


def load_valid_file_ids(infile="data/multiple_choice/valid_list.txt"):
    with open(infile, 'r') as ifh:
        rlt = [int(e.strip()) for e in ifh.readlines()]
    return rlt

def statisics_counting_time_out(inDir='_runs/data_April05_MChoice/', time_outs=[15,20, 30, 35, 40, 45, 60, 75, 90, 135, 165, 195, 225, 255, 285, 315, 345, 375, 435, 555, 1080, 1200]):
    tm_dict = dict()
    pd_dict = dict()
    for i in range(1200):
        with open(inDir+"log_data_"+str(i)+".txt", 'r') as ifh:
            "total time CL: data_999:  21.352967977523804"
            "predict False"
            for ln in ifh:
                wlst = ln.split()
                if "time" in wlst: tm_dict[i] = float(wlst[-1])
                if "predict" in wlst: pd_dict[i] = wlst[1]
    for time_out in time_outs:
        mchoice_dict = defaultdict(int)
        yn_dict = defaultdict(int)
        for j in range(10):
            for k in range(24):
                true_mchoice = 0
                for g in range(5):
                    id = j*120 + k*5 + g

                    if tm_dict[id] > 1200 and j == 4: print("id", id)

                    if tm_dict[id] < time_out:
                        true_mchoice += 1
                    else:
                        true_mchoice += 0
                if true_mchoice >= 4:
                    mchoice_dict[j+3] +=1
                else:
                    mchoice_dict[j+3] +=0
                yn_dict[j+3] += true_mchoice
        print(time_out)
        print(mchoice_dict)
        print("&".join([str(time_out)] + [str(mchoice_dict[i + 3]) for i in range(10)]))
        print(yn_dict)
        print("&".join([str(time_out)]+[str(yn_dict[i+3]) for i in range(10)]))

def load_multichoice_valid_list(ifile='data/multiple_choice/valid_list.txt'):
    with open(ifile, 'r') as ifh:
        lst = [int(e.strip()) for e in ifh.readlines()]
    # print(len(lst))
    # print(lst)
    return lst


if __name__ == '__main__':
    create_list_of_sat_cases_exp2(inFile1='data/ValidSyllogism/valid_syllogism_3.txt',
                                  inFile2='data/Syllogism/syllogism4pretrained.txt',
                                  outFile='data/Syllogism/syllogism4pretrained_with_answer.txt')
    #load_multichoice_valid_list()
    #statisics_counting_time_out()
    # find_valid_taskId()
    #create_valid_syllogistic_reasoning(4)
    # for i in range(10):
    #    get_syllogism_not_in_valid_list(i+3)
    # generate_chained_syllogism(3, opath='data/ValidSyllogism/')
    # for N in range(9):
    #    create_valid_syllogistic_reasoning(N + 4, ifile_prefix="data/ValidSyllogism/valid_syllogism_")

    # create_valid_syllogistic_reasoning()
    # get_syllogism_not_in_valid_list()

    # visualizaing_accuracy_with_num_of_epoches()
    # visualizaing_accuracy_of_family_data(ifile="data/family_result_3.txt")
    # create_table_for_family_data(ifile="data/family_data_statistics_3A.txt")
    # create_error_records(ifile0="data/family_test_detail.txt", ifile1="data/Family/cFamily_relation_test.txt")
    
    # vis_model()
    #view_dim_initR_time_2d()

    ## when we use pre-trained vectors as central points, we cannot fix them during the constructive learning process
    # ex5_get_sat_list(dir_root="/home/user/dongt/git/ENN22S/_runs/", ofile='sat_list.txt')

    # ex5_wrong_unsat_list(dir_root="/home/user/dongt/git/ENN22S/", dir_pat="_runs/Syllogism_RS0_WE_fc{}pt{}cop{}/",
    #                     sat_file='data/Syllogism/sat_list.txt')

    #ex5_refute_value_of_pre_trained_vector(sat_list_file="/home/user/dongt/git/ENN22S/data/Syllogism/sat_list.txt",
    #                                       dir_root="/home/user/dongt/git/ENN22S/runs_rlt/",
    #                                       ofile='wrong_unsat_prediction.txt')

    # ex5_which_is_the_fastest_for_sat_list

    # ex5_check_any_false(dir_root="/home/user/dongt/git/ENN22S/runs_rlt/", fix_center=True)
    # ex5_check_any_false(dir_root="/home/user/dongt/git/ENN22S/_runs/", fix_center=False)


    # ex5_error_patterns_when_fixing_center(dir_root="/home/user/dongt/git/ENN22S/_runs/")

    # ex5_find_total_runtime(data_dir="/home/user/dongt/git/ENN22S/_runs/", fix_center=True)

    #list_valid_results(data_dir="/home/user/dongt/git/ENN22S/_runs/",  dir_pattern="ValidSyllogism_V10_DIM{}COP{}RandRotNum{}R{}")

    #list_number_of_it(data_dir="/home/user/dongt/git/ENN22S/_runs/",
    #                  dir_pattern="ValidSyllogism_V10_DIM{}COP{}RandRotNum10R10")

    # get_all_word_stems()
    #create_bert_vec()
    #filter_glove_list(ifile="data/glove.6B.50d.txt", ofile="data/word_glove.6B.50d.txt", word_file="data/words.txt")
    #filter_glove_list(ifile="data/glove.6B.100d.txt", ofile="data/word_glove.6B.100d.txt", word_file="data/words.txt")
    #filter_glove_list(ifile="data/glove.6B.200d.txt", ofile="data/word_glove.6B.200d.txt", word_file="data/words.txt")
    #filter_glove_list(ifile="data/glove.6B.300d.txt", ofile="data/word_glove.6B.300d.txt", word_file="data/words.txt")