import json
import numpy as np
import os
from pprint import pprint
from prettytable import PrettyTable
import matplotlib.pyplot as plt


def evaluate_exp1(result_dir='', dim = 2, init_loc = 0):
    valid_count = 0
    result_dir = result_dir.format(dim, init_loc)
    for f in os.listdir(result_dir):
        if not f.endswith('.txt'): continue
        with open(result_dir + f, 'r') as ifh:
            for ln in ifh.readlines():
                if ln.startswith(' predict True'): valid_count += 1
    return valid_count

def get_time_cost(result_dir='', dim = 2, init_loc = 0):
    t_table = {"time4valid": [],
                "time4invalid": [],
               "time4all": []}
    result_dir = result_dir.format(dim, init_loc)
    for f in os.listdir(result_dir):
        if not f.endswith('.txt'): continue
        with open(result_dir + f, 'r') as ifh:
            " total time CL: data_3:  7.068948030471802"
            valid_t = False
            cost_t = 0.0
            for ln in ifh.readlines():
                if ln.startswith(' total time CL'):
                    segs = ln.split(':')
                    cost_t = float(segs[2].strip())
                    # if cost_t > 30: cost_t = 30
                    if cost_t == 1827.8814771175385:
                        print()
                if ln.startswith(' predict True'): valid_t = True
        if valid_t:
            t_table["time4valid"].append(cost_t)
        else:
            t_table["time4invalid"].append(cost_t)
        t_table["time4all"].append(cost_t)

    return t_table


def show_time_statistics(eval_dict, fname, il):
    values = [eval_dict[d]['time'][fname] for d in [2, 3, 15, 30, 100, 200, 1000, 2000, 3000]]
    fig, ax = plt.subplots()
    ax.boxplot(values, showfliers=False)
    ax.set_xticklabels([2, 3, 15, 30, 100, 200, 1000, 2000, 3000])

    lable = ""
    if fname == "time4valid":
        lable = "Time (in seconds) for determining no counter-example"
    elif fname == "time4invalid":
        lable = "Time (in seconds) to construct a counter-example, if exists"
    elif fname == "time4all":
        lable = "Time (in seconds) used to determine the validity"
    plt.xlabel("The dimension of spheres")
    plt.ylabel(lable)
    plt.savefig('pic/'+fname+str(il))


if __name__ == '__main__':
    eval_dict = {}
    with open('params.json', 'r') as ijh:
        params = json.load(ijh)
        data_dir = params['exp1']["data_dir"]
        log_dir = params['exp1']["output_dir"]
        dims = params['exp1']["DIMs"]
        InitLoc = params['exp1']["InitLoc"]
        t = PrettyTable(
            ['MethodID for Sphere Initialisation', 'Dimension of Sphere', 'epoch',
             '#Identified Valid Reasoning', 'Total Number'])

        for il in InitLoc:
            eval_dict[il] = {}
            for d in dims:
                exp_dir = "../data/hsphnn_runs/%s_%s/" % (data_dir.split("/")[-2], log_dir)
                valid_num = evaluate_exp1(result_dir=exp_dir, dim = d, init_loc = il)
                t.add_row([il, d, 1, valid_num, 24])
                eval_dict[il][d] = valid_num
        print(t)

        t4valid = []
        t4invalid = []
        t4all = []

        for il in InitLoc:
            eval_dict[il] = {}
            for d in dims[:]:
                exp_dir = "../data/hsphnn_runs/%s_%s/" % (data_dir.split("/")[-2], log_dir)
                valid_num = evaluate_exp1(result_dir=exp_dir, dim = d, init_loc = il)
                efficiency = get_time_cost(result_dir=exp_dir, dim = d, init_loc = il)
                eval_dict[il][d] = {"valid_num":valid_num, "time": efficiency}
                t4valid += efficiency["time4valid"]
                t4invalid += efficiency["time4invalid"]
                t4all += efficiency["time4all"]
            for fn in ["time4valid", "time4invalid", "time4all"]:
                show_time_statistics(eval_dict[il], fn, il)
            '''
            for l in [t4valid,t4invalid,t4all]:
                nl = np.array(l)
                print('time for ', nl.max(), nl.min(), nl.mean(), np.median(nl), "<120s total 216", len([s for s in l if s < 120]), "<5s total 2088", len([s for s in l if s <5]))
            '''
            print('time for valid')
            nl = np.array(t4valid)
            # pretty table
            t = PrettyTable(
                ['type of syllogism', 'number of syllogism', 'max time cost (in seconds)', 'min -',
                 'mean -', 'median -' ])
            print('time: max=', "{:0.2f}".format(float(nl.max())),'min=',  nl.min(),  'mean=', nl.mean(),'median=', np.median(nl), "<30s",
                  len([s for s in t4valid if s < 30]))
            x30 = len([s for s in t4valid if s < 30]) / len(t4valid)
            x60 = len([s for s in t4valid if s < 60]) / len(t4valid)
            x120 = len([s for s in t4valid if s < 120])/len( t4valid)
            t.add_row(['valid syllogism', 240, "{:0.2f}".format(float(nl.max())), "{:0.2f}".format(float(nl.min())), "{:0.2f}".format(float(nl.mean())), "{:0.2f}".format(float(np.median(nl)))]) #, '{:0.2}'.format(x)])
            print('time for invalid')
            nl = np.array(t4invalid)
            x = len([s for s in t4invalid if s < 20]) / len( t4invalid)
            print('time: max=', nl.max(), 'min=', nl.min(), 'mean=', nl.mean(),'median=', np.median(nl), "<5s", len([s for s in t4invalid if s <5]))
            t.add_row(['invalid syllogism', 2320, "{:0.2f}".format(float(nl.max())), "{:0.5f}".format(float(nl.min())),
                       "{:0.2f}".format(float(nl.mean())), "{:0.2f}".format(float(np.median(nl)))]) #, '{:0.2}'.format(x)])

            print('time for all')
            nl = np.array(t4all)
            x = len([s for s in t4all if s < 20]) /len( t4all)
            t.add_row(['all syllogism', 2560, "{:0.2f}".format(float(nl.max())), "{:0.5f}".format(float(nl.min())),
                       "{:0.2f}".format(float(nl.mean())), "{:0.2f}".format(float(np.median(nl)))]) #, '{:0.2}'.format(x)])

            print('time: max=', nl.max(),  'min=', nl.min(),'mean=', nl.mean(),'median=', np.median(nl), "<5s",
                  len([s for s in t4all if s < 5]), "<30s", len([s for s in t4all if s < 30]))
            print(t)
