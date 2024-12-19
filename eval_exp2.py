import json
import os
from pprint import pprint
from collections import defaultdict
from load_data import *
from utils import *
from prettytable import PrettyTable

valid_list = [0, 11, 133, 134, 141, 142, 15, 195, 203, 225, 229, 233, 237, 25, 3, 33, 34, 37, 38, 41, 45, 59, 63, 69]
unsat_list = [1, 10, 132, 135, 140, 143, 14, 194, 202, 224, 228, 232, 236, 24, 2, 32, 35, 36, 39, 40, 44, 58, 62, 68]

valid_string = """
    all S M1, all M1 P, no S P
    all S M1, all M1 P, some-not S P
    all M1 S, all M1 P, no S P
    all M1 S, all P M1, no S P
    all M1 S, some-not M1 P, all S P
    all S M1, no M1 P, all S P
    all S M1, no M1 P, some S P
    all S M1, no P M1, all S P
    all S M1, no P M1, some S P
    all M1 S, no M1 P, all S P
    all M1 S, no P M1, all S P
    all M1 S, some M1 P, no S P
    all M1 S, some P M1, no S P
    some-not S M1, all P M1, all S P
    no S M1, all P M1, all S P
    no S M1, all P M1, some S P
    no M1 S, all P M1, all S P
    no M1 S, all P M1, some S P
    some S M1, all M1 P, no S P
    some M1 S, all M1 P, no S P
    some S M1, no M1 P, all S P
    some S M1, no P M1, all S P
    some M1 S, no M1 P, all S P
    some M1 S, no P M1, all S P
"""


def generate_UNSAT_list():
    with open('params.json', 'r') as ijh:
        params = json.load(ijh)
        data_dir = params["exp3"]["data_dir"]
    d = ValidSyllogism(data_dir=data_dir, use_random_symbol=False)
    unsat_lst = []
    for i in valid_list:
        slst = d.raw_data_list[i]
        wlst = slst[-1].strip().split()
        unsat_rel = negate_syllogistic_relation(wlst[0])
        unsat_conclusion = ' '.join([unsat_rel] + wlst[1:])
        unsat = slst[:-1] + [unsat_conclusion]
        indx = d.raw_data_list.index(unsat)
        unsat_lst.append(indx)
    print('the number of unsat syllogistic reasoning: ', len(unsat_lst))
    print(unsat_lst)
    return unsat_lst


def generate_meaningful_syllogism():
    in_file1 = "data/Syllogism/syllogism4pretrained.txt"
    in_file2 = "data/ValidSyllogism/Possible_Syllogism_3.txt"
    out_file = "data/ValidSyllogism/meaningful_sllogism_3.txt"
    result_list = []

    source = []
    with open(in_file1, 'r') as ifh2:
        for ln in ifh2:
            "all S M0, all M0 P, all S P"
            sLst = [s.strip().split() for s in re.split(", |:", ln.strip())]
            source.append(sLst)

    with open(in_file2, 'r') as ifh2:
        for ln in ifh2:
            "all S M0, all M0 P, all S P"
            sLst = [s.split()[0] for s in ln.strip().split(',')]
            if len(sLst) != 3: continue
            candidates = []
            for s in source:
                if s[0][0] == sLst[0] and s[1][0] == sLst[1] and s[2][0] == sLst[2]:
                    candidates.append(s)
            sLst = re.split(', |: | ', ln.strip())
            results = []
            for scand in candidates:
                ss = scand[0] + scand[1] + scand[2]
                ids11 = sLst.index(sLst[1])
                ids12 = sLst[3:].index(sLst[1])

                idc11 = ss.index(ss[1])
                idc12 = ss[3:].index(ss[1])

                ids21 = sLst.index(sLst[2])
                ids22 = sLst[3:].index(sLst[2])

                idc21 = ss.index(ss[2])
                idc22 = ss[3:].index(ss[2])

                if ids12 == idc12 and ids22 == idc22:
                    results.append(scand)
            if len(results) == 0:
                print("**** ", ln)
            else:
                one_result = ','.join([' '.join(e) for e in random.choice(results)])
                print(one_result)




def get_failed_cases(fdir=''):
    lst = []
    for f in os.listdir(fdir):
        if not f.endswith('.txt'): continue
        with open(fdir + f, 'r') as ifh:
            for ln in ifh.readlines():
                if ln.startswith('SphNN'):
                    n = int(f.split('.')[0].split('_')[-1])
                    lst.append(n)
    return lst


def evaluate_exp3(result_dir='', stat_file = ""):
    # the original task: ['all HiWsnFC1ekq9 hObWNvvNpxop', 'all hObWNvvNpxop csFCNexY0OlZ', 'all HiWsnFC1ekq9 csFCNexY0OlZ']
    # from the above line, we know the satisfiability of this reasoning
    info_lines = []
    info_lines.append("the original task")
    info_lines.append("The accepted reply")
    info_lines.append(" self contradicts")
    info_lines.append(" update update")
    info_lines.append(" approves ")
    info_lines.append("decision is not consistent with your explanation")
    info_lines.append("Your decision is correct! Please correct your explanation")
    info_lines.append(" decision with ")
    info_lines.append(" hallucinat")
    info_lines.append("SphNN decides the task as")
    info_lines.append(" ChatGPT makes ")

    lines = []
    for f in os.listdir(result_dir):
        if not f.endswith('.txt'): continue
        infolst = [f]
        with open(result_dir + f, 'r') as ifh:
            for ln in ifh.readlines():
                for info in info_lines:
                    if info in ln: infolst.append(ln.strip())
        lines.append(':'.join(infolst))
    with open('./data/results/'+stat_file, 'w') as ofh:
        ofh.write('\n'.join(lines))


def generate_statistics_exp3(result_dir='', stat_file = ""):
    """
    :param result_dir:
    :param stat_file:
    :return:
    """
    def get_decision(info, who="", round=-1):
        SATU = info.split("SphNN decides the task as ")[1].strip().split()[0]
        if who == "sphnn":
            return SATU
        elif who == "chatgpt":
            if " update " not in info:
                if round != -1 or round != 0: return "0"
                if "ChatGPT makes correct decision" in info: return SATU
                else:
                    SATU = info.split("ChatGPT decides as ")[1].strip().split()[0].strip('.')
                    return SATU
            elif round == 0:
                if " update " in info:
                    if "The accepted reply:" in info.split(" update update")[0]:
                        seg = info.split(" update update")[0].split("The accepted reply:")[1]
                        if "cannot" in seg[:20].lower() or "incompatible" in seg[:100].lower():
                            return "UNSAT"
                        else:
                            return "SAT"
                    else:
                        print()
            else:
                if " update " not in info:
                    return "NONE"
                seg = info.split(" update update")
                if len(seg) > round:
                    seg0 = seg[round]
                    if "The accepted reply:" in seg0:
                        seg1=seg0.split("The accepted reply:")[1]
                        if "cannot" in seg1[:20].lower() or "incompatible" in seg1[:100].lower():
                            return "UNSAT"
                        else:
                            return "SAT"
                    else: return "empty"
                else:
                    return "NONE"

    def get_explanation(info, who="", round=-1):
        exp_dict = {
            "entity hallucinating": "hallu0",
            "relation hallucinating": "hallu1",
            "self contradicts": "hallu2",
            "without explanation":"empty",
            "wrong decision": "",
            "correct decision with correct relation": "expl"
        }
        if who == "sphnn": return
        if " update " not in info:
            if round == -1 or round == 0:
                for k, v in exp_dict.items():
                    if k in info: return v
            return 0
        else:
            seg = info.split(" update update")
            if len(seg) > round:
                seg0 = seg[round]
                if "The accepted reply:" in seg0:
                    seg1 = seg0.split("The accepted reply:")[1]
                    for k, v in exp_dict.items():
                        if k in seg1: return v
                return 0
            return 0

    table_numbers = {
                     "correct": {"expl": 0,"hallu0":0, "hallu1":0, "hallu2":0, "empty":0},
                     "incorrect": {"expl":0, "rest":0}
                     }
    with open(result_dir + stat_file, 'r') as ifh:
        for info in ifh.readlines():
            sphnn_decision = get_decision(info, who="sphnn")
            for i in [0, 1, 2, 3]:
                # 0: for first reply without feedback;
                # 1 for the reply with 1 feedback,
                # -1 the final replay
                chatgpt_decision = get_decision(info, who="chatgpt", round=i)
                if chatgpt_decision == "NONE":
                    break
                explanation = get_explanation(info, who="chatgpt", round=i)
                if chatgpt_decision == sphnn_decision:
                    table_numbers["correct"][explanation] +=1
                elif explanation == "expl":
                    table_numbers["incorrect"][explanation] += 1
                else:
                    table_numbers["incorrect"]["rest"] += 1

    pprint(table_numbers)
    return table_numbers


def statistic_json(result_dir="", max_num_feedback=0):
    statDic = {0: {"c_exp": 0, "c_hlu0": 0, "c_hlu1": 0, "c_hlu2": 0, "i_exp": 0, "i_rest": 0, "c_ratio":0, "irr_ratio":0},
               "F": {"c_exp": 0, "c_hlu0": 0, "c_hlu1": 0, "c_hlu2": 0, "i_exp": 0, "i_rest": 0, "c_ratio":0, "irr_ratio":0}}
    roundDic = defaultdict(int)
    for f in os.listdir(result_dir):
        if not f.endswith('.json'): continue
        t = PrettyTable(
            ['#Correct+Expl', '#HALLU 0', '#HALLU 1', '#HALLU 2', '#Inorrect+Expl', '#Rest', '#Ratio of Correct+Expl', '#Irrationality ratio'])
        with open(result_dir + f, 'r') as ijh:
            result = json.load(ijh)
            if result["0"]["d_type"] == True and result["0"]["expl_type"].lower() == "expl":
                statDic[0]["c_exp"] += 1
            elif result["0"]["d_type"] == True and result["0"]["expl_type"].lower() == "hlu0":
                statDic[0]["c_hlu0"] += 1
                statDic[0]["irr_ratio"] += 1
            elif result["0"]["d_type"] == True and result["0"]["expl_type"].lower() == "hlu1":
                statDic[0]["c_hlu1"] += 1
                statDic[0]["irr_ratio"] += 1
            elif result["0"]["d_type"] == True and result["0"]["expl_type"].lower() == "hlu2":
                statDic[0]["c_hlu2"] += 1
                statDic[0]["irr_ratio"] += 1
            elif result["0"]["d_type"] == False and result["0"]["expl_type"].lower() == "expl":
                statDic[0]["i_exp"] += 1
                statDic[0]["irr_ratio"] += 1
            elif result["0"]["d_type"] == False and result["0"]["expl_type"].lower() != "expl":
                statDic[0]["i_rest"] += 1
            else:
                pprint(result)

            finalRound = str(result["sphnn_feedback_num"])
            if result[finalRound]["d_type"] == True and result[finalRound]["expl_type"].lower() == "expl":
                roundDic[result["sphnn_feedback_num"]] += 1
                statDic["F"]["c_exp"] += 1
            elif result[finalRound]["d_type"] == True and result[finalRound]["expl_type"].lower() == "hlu0":
                statDic["F"]["c_hlu0"] += 1
                statDic["F"]["irr_ratio"] += 1
            elif result[finalRound]["d_type"] == True and result[finalRound]["expl_type"].lower() == "hlu1":
                statDic["F"]["c_hlu1"] += 1
                statDic["F"]["irr_ratio"] += 1
            elif result[finalRound]["d_type"] == True and result[finalRound]["expl_type"].lower() == "hlu2":
                statDic["F"]["c_hlu2"] += 1
                statDic["F"]["irr_ratio"] += 1
            elif result[finalRound]["d_type"] == False and result[finalRound]["expl_type"].lower() == "expl":
                statDic["F"]["i_exp"] += 1
                statDic["F"]["irr_ratio"] += 1
            elif result[finalRound]["d_type"] == False and result[finalRound]["expl_type"].lower() != "expl":
                statDic["F"]["i_rest"] += 1
            else:
                print()

    #pprint(statDic)
    statDic[0]["c_ratio"] = statDic[0]["c_exp"] / 256
    statDic[0]["irr_ratio"] = statDic[0]["irr_ratio"] / 256
    t.add_row([statDic[0][e] for e in list(statDic[0].keys())])
    print("no feedback")
    print(t)

    statDic["F"]["c_ratio"] = statDic["F"]["c_exp"] / 256
    statDic["F"]["irr_ratio"] = statDic["F"]["irr_ratio"] / 256
    t = PrettyTable(
        ['#Correct+Expl', '#HALLU 0', '#HALLU 1', '#HALLU 2', '#Inorrect+Expl', '#Rest', '#Ratio of Correct+Expl', '#Irrationality ratio'])
    t.add_row([statDic["F"][e] for e in list(statDic["F"].keys())])
    print("with maximum {} times feedback".format(max_num_feedback))
    print(t)
    #pprint(roundDic)
    #'pprint(roundDic)

    t = PrettyTable(
        ['#Num of feedback', *range(max_num_feedback+1)])
    t.add_row(['#Tasks with correct decision and explanation']+[roundDic[e] for e in range(max_num_feedback+1)])
    print("Number of correct decision and correct explanation vs. the number of feedbacks")
    print(t)


def initiate_dictionary(valid_str):
    con_sis_Dic = {}
    plst = valid_str.strip().split("\n")
    for aplst in plst:
        p1, p2, q = aplst.split(",")
        p1 = p1.replace("1", "0")
        p2 = p2.replace("1", "0")
        k_premise = ";".join([p1, p2]).strip()
        if k_premise not in con_sis_Dic.keys():
            con_sis_Dic[k_premise] = {"sat": {}, "unsat": {}}
            con_sis_Dic[k_premise]["sat"] = {q: ""}
        else:
            con_sis_Dic[k_premise]["unsat"] = {q: ""}
    return con_sis_Dic


def find_conclusion_consistency_conflict(result_dir="", valid_str=""):
    """"raw": ["some-not S M0", "all P M0", "all S P"]"""
    con_sis_Dic = initiate_dictionary(valid_str)
    valid_premises = list(con_sis_Dic.keys())
    for f in os.listdir(result_dir):
        if not f.endswith('.json'): continue
        with open(result_dir + f, 'r') as ijh:
            result = json.load(ijh)
            raw = result["task"]["raw"]
            k_premise = ";".join(raw[:-1])
            k_c = raw[-1]
            if k_premise in valid_premises:
                comp = con_sis_Dic[k_premise]
                if k_c in comp["sat"].keys():
                    N = result["sphnn_feedback_num"]
                    for i in range(N):
                        if result[str(i)]["d_type"] != True:
                            comp["sat"][k_c] = "UNSAT"
                else:
                    N = result["sphnn_feedback_num"]
                    for i in range(N):
                        if result[str(i)]["d_type"] == True:
                            comp["unsat"][k_c] = "SAT"
                con_sis_Dic[k_premise] = comp
            else:
                if k_premise not in con_sis_Dic.keys():
                    con_sis_Dic[k_premise] = {}
                comp = con_sis_Dic[k_premise]
                N = result["sphnn_feedback_num"]
                for i in range(N):
                        if result[str(i)]["d_type"] == True:
                            comp[k_c] = "SAT"
                con_sis_Dic[k_premise] = comp
    #pprint(con_sis_Dic)


if __name__ == '__main__':
    eval_dict = {}
    with open('params.json', 'r') as ijh:
        params = json.load(ijh)
        for tp in ['all_3tb','all_4o','all_4o10F']:
            print("*"*100)
            if tp == 'all_3tb': print("using GPT-3.5-turbo, maximum 2 time feedback")
            elif tp == 'all_4o': print("using GPT-4o, maximum 2 time feedback")
            elif tp == 'all_4o10F': print("using GPT-4o, maximum 10 time feedback")
            for block in params["exp2"][tp]:
                    data_dir = block["output_dir"]
                    print(data_dir)
                    statistic_json(result_dir=data_dir, max_num_feedback=block["max_num"])
