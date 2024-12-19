#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  File name:    run_3.py
"""

import json
import argparse

from load_data import *
from models_3 import *
from utils import *

from torch.utils.tensorboard import SummaryWriter
import logging

torch.set_default_dtype(torch.float64)
maxCount = 1000


class Experiment:
    def __init__(self, args):
        self.save_dir = "../data/hsphnn_runs/%s_%s" % (args.data_dir.split("/")[-2], args.log)
        self.args = args

    def jointly_train_and_eval(self):
        results = []

        if self.args.pre_train == 1:
            dfile = "data/word_glove.6B.50d.txt"
            dim, dic = embed_dict(data_file=dfile, norm=self.args.norm_vec)
        elif self.args.pre_train == 2:
            dfile = "data/wstem_bert_embeddings.txt"
            dim, dic = embed_dict(data_file=dfile, norm=self.args.norm_vec)
        elif self.args.pre_train == 3:
            dfile = "data/gpt35_embeddings.txt"
            dim, dic = embed_dict(data_file=dfile, norm=self.args.norm_vec)
        else:
            dim = self.args.dim
            dic = dict()

        for i, s in enumerate(d.data_list):
            if i < self.args.start:                     continue
            elif i - self.args.start >= self.args.num:  return

            #valid_list = [0, 11, 133, 134, 141, 142, 15, 195, 203, 225, 229, 233, 237, 25, 3, 33, 34,
            #              37, 38, 41, 45, 59, 63, 69]

            fw = None
            if self.args.save:
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
                fw = open(self.save_dir + "/log_data_{}.txt".format(i), "w")

            start_time = time.time()
            model = HyperbolicSphNN(args.cuda, self.args.lr, self.args.wd, self.args.save, None, fw,
                        timer=self.args.timer, diagram=self.args.counter_example, save_dir = self.save_dir,
                        ent_num=s[0], input_statement = s[1], raw_data = d.raw_data_list[i], entdic=d.id2ent_dict_list[i],
                        pre_train=self.args.pre_train, dim=dim, eps=self.args.eps, epoch=self.args.epoch,
                        cop=0, rand_rotate_num=360, center_fixed=args.fix_center,
                        init_loc=self.args.init_loc, init_r=self.args.ref_R, pre_train_dic=dic).to(device)

            opt = optim.SGD(params=model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
            optBig = optim.SGD(params=model.parameters(), lr=self.args.lr *1000, weight_decay=self.args.wd)
            opt100 = optim.SGD(params=model.parameters(), lr=self.args.lr*100, weight_decay=self.args.wd)
            optInit = optim.SGD(params=model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)

            logger.info(model)
            logger.info(opt)

            model.opt = opt
            model.optInit = opt
            model.opt100 = opt100
            model.optBig = optBig

            if self.args.save:
                fw = open(self.save_dir + "/log_data_{}.txt".format(i), "w")
                model.save_spheres_to_file(description="init sphere ")

            # Train
            logger.info("start jointly training data: " + str(i))

            ent_num = s[0]
            if self.args.sat:
                eval_flag='sat'
                statement_set = list(s[1][:3])
                model.set_criteria(ent_num, statement_set, d.id2ent_dict_list[i], logger)
            elif self.args.valid:
                eval_flag = 'valid'
                statement_set = list(s[1])
                statement_set = negate_conclusion(statement_set)
                # check unsatisfiability
                model.set_criteria(ent_num, statement_set, d.id2ent_dict_list[i], logger)

            # Save
            if self.args.save:
                # Save embeddings
                torch.save(model.state_dict(), self.save_dir + "/init_model_data_%s.pt" % (str(i)))
                logger.info("model saved!")

            # try to reach the global loss of zero, if reached and needs a counter example
            loss = model.constructive_learning4chain_algo(i, logger)
            if loss == 0 and model.dim == 2:
                model.construct_diagram(i, logger)

            result = model.evaluate_constructed_diagrams(i, gloss=loss, flag=eval_flag, logger=logger)
            results.append(result)

        # Overall result
        if len(results) != 0:
            valid_lst = [r for r in results if r == 'True']
            logger.info("overall acc:  {} / {}".format(len(valid_lst), len(results)))
            logger.info("total time:  {}".format(time.time() - start_time))

        if self.args.save:
            model.save_spheres_to_file(description="final spheres ")
            model.fw.write("overall acc:  {} / {}".format(len(valid_lst), len(results)))
            model.fw.write("total time:  {}".format(time.time() - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data/Syllogism/", required=False, help="input dataset file directory, ('data/Syllogism/', ...)")
    parser.add_argument("--data", type=str, default="", required=False, help="input dataset file")

    parser.add_argument("--cuda", action="store_true", default=True, help="whether to use cuda or not")
    parser.add_argument("--seed", type=int, default=2020, help="random seed")
    parser.add_argument("--log", type=str, default="tensorboard_log", nargs="?", help="where to save the log")

    # three types of tasks
    parser.add_argument("--valid", action="store_true", default=True,
                        help="whether inputs statements are valid")
    parser.add_argument("--sat", action="store_true", default=False,
                        help="whether inputs statements are satisfiable")
    parser.add_argument("--multiple_choice", action="store_true", default=False,
                        help="choose the valid one from five choices")

    parser.add_argument("--norm_vec", action="store_true", default=True,
                        help="normalize pre_trained vectors")

    parser.add_argument("--t_num", type=int, default=1, help="number of tasks")
    parser.add_argument("--start", type=int, default=0, help="start (for slicing)")
    parser.add_argument("--num", type=int, default=999999999, help="num (for slicing)")
    parser.add_argument("--n", action="store_true", default=False, help="whether to del logdir")

    parser.add_argument("--epoch", type=int, default= 1, help="number of epochs to train")
    parser.add_argument("--timer", type=int, default=20, help="timer (in minutes) for each task")
    parser.add_argument("--counter_example", action="store_true", default=True, help="whether show one counter example")

    parser.add_argument("--save", action="store_true", default=False, help="whether to save log, results and embeddings")
    parser.add_argument("--vis", action="store_true", default=False, help="whether to vis in dim == 2 or 3")

    parser.add_argument("--init_loc", type=int, default=0,
                        help="0: balls are initialised co-inside; 1: balls are initialised uniformly located on a sphere/circle; 2: using pre-trained vector")
    parser.add_argument("--pre_train", type=int, default=0,
                        help="whether use pre-trained value for initialization, 0 for no, 1 for GloVE, 2 for BERT, 3 for GPT3.5")
    parser.add_argument("--ref_R", type=int, default=1,
                        help="Euler diagrams are initially located at the sphere of a reference ball. ref_R is the radius of this reference ball.")
    parser.add_argument("--dim", type=int, default=5, help="dim (real dim = dim+1)")

    parser.add_argument("--fix_center", type=int, default=0, help="initial central points are fixed, 1: fix center -- too strong to be feasible, 2: fix the orientation of the center; 0: not fixed")

    parser.add_argument("--eps", type=float, default=1e-2, help="eps") #changed from 0, 1e-9

    parser.add_argument("--lr", type=float, default=0.0001, help="initial learning rate")
    parser.add_argument("--wd", type=float, default=0, help="weight decay (L2 loss on parameters)")
    parser.add_argument("--dr", type=float, default=0, help="decay rate of lr")
    parser.add_argument("--rs", action="store_true", default=True, help="whether replace S, M, P with random symbols")

    args = parser.parse_args()

    with open('params.json') as parms_file:
        params = json.load(parms_file)
        log_dir = params["output_dir"]+"%s_%s" % (args.data_dir.split("/")[-2], args.log)
    print('log_dir', log_dir)

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    writer = SummaryWriter(log_dir)
    logger.info(args)

    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # Load Data
    if args.sat:
        d = SyllogismData(data_dir=args.data_dir)
    elif args.valid:
        d = ValidSyllogism(data_dir=args.data_dir, use_random_symbol=args.rs)
    else:
        raise NotImplementedError("unknown data_dir:" + args.data_dir)
    logger.info(d)

    experiment = Experiment(args=args)

    t_total = time.time()
    experiment.jointly_train_and_eval()

    logger.info("optimization finished!")
    logger.info("total time elapsed: {:.4f} s".format(time.time() - t_total))
