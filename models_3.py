#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from itertools import permutations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import time
from utils import *
import vis_util

torch.set_default_dtype(torch.float64)
SHOW_DIAGRAM = False
USE_BIG_OPT = 2
USE_SMALL_OPT = 0.02
ZERO_RADIUS_SIG = np.inf   # unsatisfiable configuration may force a radius into zero during the construction process


class HyperbolicSphNN(torch.nn.Module):
    def __init__(self, cuda, lr, w_decay, save, OPT, fw, timer= 5*60, diagram=False, save_dir='',
                 ent_num=3, input_statement = [], raw_data = [], entdic=None,
                 pre_train=None, dim=2, eps=0, epoch=1, cop=0,
                 rand_rotate_num=360, center_fixed=0, p=2, init_loc=0, init_r = 50, pre_train_dic=dict()):

        super(HyperbolicSphNN, self).__init__()
        self.sphnn_timer = time.time()
        self.task_timer = time.time()
        self.task_duration = timer * 60
        self.OUT_OF_TIME = False
        self.diagram = diagram
        self.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.w_decay = w_decay
        self.save = save
        self.save_dir = save_dir
        if save: self.fw = fw
        else: self.fw = False

        self.opt = OPT
        self.opt100 = OPT
        self.optBig = OPT
        self.optInit = OPT

        self.TB_origin, self.TB_0 = [], []
        self.cop = cop
        self.ent_num = ent_num
        self.inputs = input_statement
        self.raw_input = raw_data
        self.ent_dict = entdic
        self.dim = dim
        self.eps = eps
        self.epoch = epoch
        self.pi = 3.14159262358979323846
        self.rand_rotate_num = rand_rotate_num
        self.center_fixed = center_fixed
        self.init_r = init_r
        self.init_loc = init_loc

        self.zero = 1e-7
        self.NUM_OF_ATTEMPTS = 1

        self.perm_set = list(permutations(range(self.dim), 2))
        self.p_q_Chi = 0
        self.k_Chi = 1
        self.g_count = []
        self.can_rotate = True
        self.g_loss_dict = {"gloss": np.inf, "value": None, "iter": 0}
        self.his_states = defaultdict()
        self.his_config = defaultdict()
        self.flag_loss = defaultdict()
        self.p = p

        self.pre_train = pre_train
        self.w2v_dict = dict()
        if self.pre_train in [1, 2, 3]:
            # using GloVE BERT GPT3.5
            self.w2v_dict = pre_train_dic
            self.dim = len(self.w2v_dict['entity'])
            self.init_loc == 2

        self.embeddings = nn.Embedding(self.ent_num, self.dim + 1)
        if self.init_loc == 1:
            nn.init.zeros_(self.embeddings.weight[:, -1])
            with torch.no_grad():
                while True:
                    nn.init.uniform_(self.embeddings.weight[:,:-1])
                    len_vec = self.embeddings.weight[:,:-1].sum(dim=1)
                    if len((len_vec == 0).nonzero()) == 0:
                        len_vec = torch.norm(self.embeddings.weight[:,:-1], dim=1) / self.init_r
                        self.embeddings.weight[:,:-1] = torch.diag(torch.reciprocal(len_vec)) @ self.embeddings.weight[:,:-1]
                        self.embeddings.weight = torch.nn.Parameter(self.embeddings.weight, requires_grad=True)
                        break

        elif self.init_loc == 0:
            nn.init.zeros_(self.embeddings.weight[:, :])
            with torch.no_grad():
                for i in range(self.ent_num):
                    self.embeddings.weight[i, -1] = 0.3
                while True:
                    nn.init.uniform_(self.embeddings.weight[:, :-1])
                    len_vec = torch.norm(self.embeddings.weight[:, :-1], p=self.p, dim=1)
                    if len((len_vec == 0).nonzero()) == 0:
                        len_vec = torch.norm(self.embeddings.weight[:, :-1], dim=1) * 10
                        self.embeddings.weight[:, :-1] = torch.diag(torch.reciprocal(len_vec)) @ self.embeddings.weight[
                                                                                                 :, :-1]
                        for i in range(self.ent_num - 1):
                            self.embeddings.weight[i, :-1] = self.embeddings.weight[self.ent_num - 1, :-1]
                        self.embeddings.weight = torch.nn.Parameter(self.embeddings.weight, requires_grad=True)
                        break

        elif self.init_loc == 2 : # using pre-trained vector
            nn.init.zeros_(self.embeddings.weight[:, -1])
            config = []
            for i in range(self.ent_num):
                wd = self.ent_dict[i].split('.')[0]
                wv = self.w2v_dict.get(wd, [0]*self.dim)
                config.append(wv + [0])
            with torch.no_grad():
                self.embeddings.weight.copy_(torch.FloatTensor(config))
                self.embeddings.weight = torch.nn.Parameter(self.embeddings.weight, requires_grad=True)
        else:
            # for debugging
            nn.init.uniform_(self.embeddings.weight[:, :])
            config = [ ( 3.3766969489408933,0.2923343089495287,0.3038184353896095,1.0126612331556524,3.52060647502747 ,  1.0),
                       ( 3.3766969489408933,0.2923343089495287,0.3038184353896095,1.0126612331556524,3.52060647502747 ,  1.0),
                       ( 3.3766969489408933,0.2923343089495287,0.3038184353896095,1.0126612331556524,3.52060647502747 ,  1.0),
                       ( 3.3766969489408933,0.2923343089495287,0.3038184353896095,1.0126612331556524,3.52060647502747 ,  1.0)]
            with torch.no_grad():
                self.embeddings.weight.copy_(torch.FloatTensor(config))
                self.embeddings.weight = torch.nn.Parameter(self.embeddings.weight, requires_grad=True)

        self.detach_list = []

        self.routes = {"D": {"D": "DONE",
                             "PO2": "INC_DIS", # increase dis INC_DIS
                             "PO1": "INC_DIS_DEC_R",   # increase dis or reduce r
                             "PP": "INC_DIS",    # increase dis
                             "PPbar": "INC_DIS_DEC_R", # increase dis or reduce r
                             "EQ": "INC_DIS"        # increase dis
                             },

                       "P": {"D": "DEC_DIS",        # reduce dis
                             "PO1": "DEC_DIS",   # reduce dis
                             "PO2": "DEC_DIS_DEC_R",     # reduce dis or reduce r
                             "P": "DONE",
                             "PPbar": "DEC_R"   # reduce r
                        },

                       "Pbar": {"D": "DEC_DIS",        # reduce dis
                                "PO3": "INC_R",   # increase r
                                "PO4": "DEC_DIS_INC_R",  # reduce dis or increase r
                                "PP": "INC_R",     # increase r
                                "Pbar": "DONE"
                                },

                       "PO": {"D": "DEC_DIS",  # reduce dis
                              "PO": "DONE",
                              "P": "INC_DIS_INC_R",  # reduce dis or increase r
                              "Pbar": "INC_DIS_DEC_R"
                            },

                       "-D": {"D": "DEC_DIS_INC_R",
                              "ND": "DONE"
                              },

                       "-P": {"NP": "DONE",
                              "P": "INC_DIS_INC_R"
                              },

                       "-Pbar": {"Pbar": "INC_DIS_DEC_R",
                                 "NPbar": "DONE"
                                 },
                       }

    def re_initialise_spheres(self):

        if self.init_loc == 1:
            nn.init.zeros_(self.embeddings.weight[:, -1])
            while True:
                nn.init.uniform_(self.embeddings.weight[:, :-1])
                len_vec = self.embeddings.weight[:, :-1].sum(dim=1)
                if len((len_vec == 0).nonzero()) == 0:
                    len_vec = torch.norm(self.embeddings.weight[:, :-1], dim=1) / self.init_r
                    self.embeddings.weight[:, :-1] = torch.diag(torch.reciprocal(len_vec)) @ self.embeddings.weight[:,
                                                                                             :-1]
                    self.embeddings.weight = torch.nn.Parameter(self.embeddings.weight, requires_grad=True)
                    break
        elif self.init_loc == 0:
            nn.init.zeros_(self.embeddings.weight[:, :])
            with torch.no_grad():
                for i in range(self.ent_num):
                    self.embeddings.weight[i, -1] = 0.3
                while True:
                    nn.init.uniform_(self.embeddings.weight[:, :-1])
                    len_vec = torch.norm(self.embeddings.weight[:, :-1], p=self.p, dim=1)
                    if len((len_vec == 0).nonzero()) == 0:
                        len_vec = torch.norm(self.embeddings.weight[:, :-1], dim=1) * 10
                        self.embeddings.weight[:, :-1] = torch.diag(torch.reciprocal(len_vec)) @ self.embeddings.weight[
                                                                                                 :, :-1]
                        for i in range(self.ent_num - 1):
                            self.embeddings.weight[i, :-1] = self.embeddings.weight[self.ent_num - 1, :-1]
                        self.embeddings.weight = torch.nn.Parameter(self.embeddings.weight, requires_grad=True)
                        break

        elif self.init_loc == 2:  # using pre-trained vector
            nn.init.zeros_(self.embeddings.weight[:, -1])
            config = []
            for i in range(self.ent_num):
                wd = self.ent_dict[i].split('.')[0]
                wv = self.w2v_dict.get(wd, [0] * self.dim)
                config.append(wv + [0])
            with torch.no_grad():
                self.embeddings.weight.copy_(torch.FloatTensor(config))
                self.embeddings.weight = torch.nn.Parameter(self.embeddings.weight, requires_grad=True)
        else:
            # for debugging
            nn.init.uniform_(self.embeddings.weight[:, :])
            config = [(1.2498741149902344, 4.841261863708496, 1.0),
                      (2.7835397720336914, 6.188019752502441, 1.0100501668584043),
                      (3.0321903228759766, 5.326048851013184, 1.9071809079471955),
                      (1.9967536926269531, 6.384208679199219, 1.0100501668584043),
                      (0.4886219907146511, 5.420104052916399, 0.7416279765959022),
                      (0.10781183099875569, 5.544401991312471, 0.3410161023629019)]
            with torch.no_grad():
                self.embeddings.weight.copy_(torch.FloatTensor(config))
                self.embeddings.weight = torch.nn.Parameter(self.embeddings.weight, requires_grad=True)

        opt = optim.SGD(params=self.parameters(), lr=self.lr, weight_decay=self.w_decay)
        optBig = optim.SGD(params=self.parameters(), lr=self.lr * 1000, weight_decay=self.w_decay)
        opt100 = optim.SGD(params=self.parameters(), lr=self.lr * 100, weight_decay=self.w_decay)
        optInit = optim.SGD(params=self.parameters(), lr=self.lr, weight_decay=self.w_decay)

        self.opt = opt
        self.optInit = opt
        self.opt100 = opt100
        self.optBig = optBig

    def re_start_the_process(self):
        self.re_initialise_spheres()

    def set_criteria(self, s0, statements, dlable, logger):
        self.TB_0, self.statement_set = create_TB_0(s0, statements)
        logger.info("TB_0 init: {}".format(self.TB_0))
        self.label_dict = dlable
        if len([e for e in self.TB_0[0]+self.TB_0[1]+self.TB_0[2] if len(e.strip())> 0]) < 3:
            return False
        return True

    def acquire_entity_relations(self, answer_data, logger):
        FLU1 = False
        count = 0
        for e1, e2, rel, _ in answer_data:
            if self.TB_origin[e1][e2] != '':
                t_rel = self.TB_origin[e1][e2]
            else:
                t_rel = inverse_relation(self.TB_origin[e2][e1])
            c_rel = rel
            if t_rel != '': count = count + 1
            if not consistent_rel(c_rel, t_rel):
                FLU1 = True
        return FLU1, self.set_criteria(self.ent_num, answer_data, self.ent_dict, logger)

    def get_target_rel(self, i, j):
        tar_state = self.TB_0[i][j]
        if tar_state == '':
            tar_state = inverse_relation(self.TB_0[j][i])
        return tar_state

    def make_qq(self, i, j):
        rel_i2j = self.get_target_rel(i, j)

        return [torch.LongTensor([i]).to(self.device), torch.LongTensor([j]).to(self.device), rel_i2j]

    def determine_syllogistic_relation(self, not_be='', can_be=''):
        if can_be in ['D', 'P', 'Pbar']:
            return can_be
        elif can_be in ['PO']:
            if not_be in ['D']:
                return '-D'
            elif not_be in ['P']:
                return '-P'
            elif not_be in ['Pbar']:
                return '-Pbar'
            else:
                raise NotImplementedError("not be?", not_be, "can be? ", can_be)

    def show_sphere(self):
        weights = self.embeddings.weight.detach().cpu().numpy()  # .tolist()
        vis_util.show_spheres(weights=weights)

    def constraint_optimisation_of_sphere(self, i, opt_sphere=0, ref_sphere=0, constraint_sphere=0, logger=None):
        #loss = self.update_single_sphere(constraint_sphere, ref_id=ref_sphere)
        loss = self.update_single_sphere(opt_sphere, ref_id=constraint_sphere)
        #print('constraint_optimisation_of_sphere', opt_sphere, ref_sphere)
        #qsr0 = self.inspect_QSR4syllogism(0, 2)
        #print('qsr0', qsr0)

        #print('in constraint_optimisation_of_sphere', opt_sphere, ref_sphere, constraint_sphere)
        loss = self.constraint_optimisation_with_random_rotation(i, opt_sphere, ref_sphere, constraint_sphere,
                                                                     logger=logger)

        #print('after constraint_optimisation_of_sphere', opt_sphere, ref_sphere)
        #qsr0 = self.inspect_QSR4syllogism(0, 2)
        #print('qsr0', qsr0)

        if loss > 0:
            #loss = self.update_single_sphere(constraint_sphere, ref_id=ref_sphere)
            loss = self.update_single_sphere(opt_sphere, ref_id=ref_sphere)
            #print('in constraint_optimisation_of_sphere', opt_sphere, constraint_sphere, ref_sphere)
            loss = self.constraint_optimisation_with_random_rotation(i, opt_sphere, constraint_sphere, ref_sphere,
                                                                     logger=logger)
        return loss

    def inspect_QSR4syllogism(self, N_i, N_j):
        """
        :param N_i:
        :param N_j:
        :return:
        """
        qq = self.make_qq(N_i, N_j)
        w = self.embeddings(qq[0])
        v = self.embeddings(qq[1])
        o_w, r_w = 0.9*torch.sin(w[:, :self.dim]), torch.sin(w[:, self.dim:])**2
        o_v, r_v = 0.9*torch.sin(v[:, :self.dim]), torch.sin(v[:, self.dim:])**2
        if self.D(o_w, r_w, o_v, r_v):
            return "D"
        elif self.PP(o_w, r_w, o_v, r_v):
            return "P"
        elif self.PPbar(o_w, r_w, o_v, r_v):
            return "Pbar"
        elif self.PO(o_w, r_w, o_v, r_v):
            return "PO"
        elif self.EQ(o_w, r_w, o_v, r_v):
            return "EQ"

    def update_single_sphere(self, N_j, ref_id=0):
        #print('update_single_sphere', N_j, ref_id)
        #qsr0 = self.inspect_QSR4syllogism(0, 2)
        #print('qsr0', 0, 2, qsr0)

        qq = self.make_qq(N_j, ref_id)
        opt_fun = self.get_opt_fun(qq)
        #print('opt fun', opt_fun)
        self.detach_list = [ref_id]

        if opt_fun != "DONE" and self.centre_not_fixed():
            self.break_from_EQ_by_vibration(qq, N_j, degree=1)
            opt_fun = self.get_opt_fun(qq)
            # print('after break EQ, opt fun', opt_fun)
            self.detach_list = [ref_id]

        last_loss = np.inf
        last_opt_fun = None

        while opt_fun !="DONE":
            #if opt_fun != last_opt_fun:
                #print('in loop, opt fun', opt_fun)
            cur_state, tar_state = self.get_state(qq, target=qq[2])
            # print(cur_state, tar_state)
            loss = self.forward(qq)
            # print(qq, loss, opt_fun)
            #if loss == ZERO_RADIUS_SIG:
            #    break
            if self.orientation_fixed() and last_loss - loss < 0 and last_opt_fun == opt_fun:
                print('break here', last_loss, loss, opt_fun)
            #    cur_state, tar_state = self.get_state(qq, target=qq[2])
            #    print('opt_fun = self.get_opt_fun(qq)', cur_state, tar_state)
            #    print('inspect relation', qq, self.inspect_QSR4syllogism(qq[0], qq[1]))
                break
            else:
                self.opt = optim.SGD(params=self.parameters(), lr=self.lr, weight_decay=self.w_decay)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            last_loss = loss
            last_opt_fun = opt_fun
            opt_fun = self.get_opt_fun(qq)
            #cur_state, tar_state = self.get_state(qq, target=qq[2])
            #print('opt_fun = self.get_opt_fun(qq)', cur_state, tar_state)
            #print('inspect relation', qq, self.inspect_QSR4syllogism(qq[0],qq[1]))

        if self.save:
            self.save_spheres_to_file(description="{} {} {}".format(N_j, ref_id, self.get_target_rel(N_j, ref_id)))

        return self.get_loss(N_j, ref_id)

    def get_triangle_status(self, i,j,k):
        q1, q2, q3 = self.make_qq(i, j), self.make_qq(j, k), self.make_qq(k, i)
        s1, _ = self.get_state(q1, target=self.get_target_rel(i, j))
        s2, _ = self.get_state(q2, target=self.get_target_rel(j, k))
        s3, _ = self.get_state(q3, target=self.get_target_rel(k, i))
        return '_'.join([s1,s2,s3])

    def get_triangle_loss(self, i, j, k):
        loss3 = self.get_loss(i,j) + self.get_loss(j, k) + self.get_loss(k, i)
        return loss3

    def update_two_spheres_fixing_ref_ori(self, N_k, N_i, ref_id=2):
        loss_ori3 = np.inf
        self.his_states = defaultdict()
        self.flag_loss = defaultdict()
        while loss_ori3 > 0:
            if self.out_of_time():
                self.OUT_OF_TIME = True
                break
            self.opt.zero_grad()
            loss_ori3 = self.loss_fixing_ori3(N_k, N_i, ref_id)
            if loss_ori3 == 0:
                break
            loss_ori3.backward()
            self.opt.step()

            # if the relation between id3 and fixed_id is broken, repair it
            #loss = self.update_single_sphere(N_k, ref_id=ref_id)
            loss_ori3 = self.get_triangle_loss(N_k, N_i, ref_id)
            states3 = self.get_triangle_status(N_k, N_i, ref_id)

            if states3 in self.his_states.keys():  # compare with the best loss in the history
                if self.his_states[states3] > loss_ori3:
                    self.his_states[states3] = loss_ori3
                elif self.his_states[states3] <= loss_ori3:
                    self.flag_loss[states3] += 1

            else:  # initialise history
                self.his_states[states3] = loss_ori3
                self.flag_loss[states3] = 0

            if self.flag_loss[states3] > self.NUM_OF_ATTEMPTS: break

        return loss_ori3

    def out_of_time(self):
        if self.task_timer > time.time():
            return False
        else:
            return True

    def centre_not_fixed(self):
        if self.center_fixed == 0:
            return True
        else:
            return False

    def orientation_fixed(self):
        if self.center_fixed == 2:
            return True
        else:
            return False

    def construct_diagram(self, i, logger):
        for N_k in range(self.ent_num-2):
            N_i = N_k + 1
            N_j = N_i + 1
            self.detach_list = [0, N_i]
            if len(self.TB_0[0][N_j])!= 0:
                loss = self.constraint_optimisation_with_random_rotation(i, N_j, N_i, 0, logger=logger)
                if loss > 0:
                    self.update_single_sphere(N_j, ref_id=N_i)
                    loss = self.constraint_optimisation_with_random_rotation(i, N_j, 0, N_i, logger=logger)
            else:
                self.update_single_sphere(N_j, ref_id=N_i)

        N_i, N_j = self.ent_num -1, self.ent_num
        if self.ent_num > 3:
            N_i, N_j = N_i - 1, N_j - 1
            while N_i > 1:
                if len(self.TB_0[N_i][0])!= 0:
                    loss = self.constraint_optimisation_of_sphere(i, opt_sphere= N_i, ref_sphere=N_j,
                                                                  constraint_sphere=0,
                                                                  logger=logger)
                else:
                    self.update_single_sphere(N_i, ref_id=N_j)

                N_i, N_j = N_i - 1, N_j - 1

            # the last triangle relation
        if loss > 0 and self.get_target_rel(0, 2) != '':
            self.update_single_sphere(2, ref_id=0)  # backward
            self.update_single_sphere(1, ref_id=2)  # backward
            loss = self.constraint_optimisation_of_sphere(i, opt_sphere=1, ref_sphere=2, constraint_sphere=0,
                                                          logger=logger)

        g_loss = self.global_loss_chain()
        if g_loss == 0:
            weights = self.embeddings.weight.detach().cpu().numpy()
            vis_util.show_spheres(weights=weights, show=False, savefile=self.save_dir + "/diagram_{}.png".format(i))
        if self.fw:
            self.save_spheres_to_file("the final configuration of backward phase only to check validity!")

    def reverse_update(self, start=0):
        ref_num = start
        opt_num = start - 1
        detach_lst = self.detach_list
        while opt_num >= 0:
            self.detach_list = [ref_num]
            loss = self.update_single_sphere(opt_num, ref_id=ref_num)
            ref_num -= 1
            opt_num -= 1
        self.detach_list = detach_lst
        return loss

    def fix_one_and_optimise_two(self, N_k, N_i, fix_id = 2):
        if self.centre_not_fixed():
            self.update_single_sphere(N_k, ref_id=fix_id)
            self.update_single_sphere(N_i, ref_id=fix_id)
        else:
            # the orientations of N_k, and fix_id are fixed
            self.update_single_sphere(N_k, ref_id=fix_id)
            # all three orientations of N_k, N_i, fix_id are fixed
            self.update_two_spheres_fixing_ref_ori(N_k, N_i, ref_id=fix_id)

    def construct4chain_n(self, data_i, chain_length, logger):
        if chain_length < 3: return
        ##
        ## forward construction
        ##
        for N_k in range(chain_length - 1):
            if self.out_of_time():
                self.OUT_OF_TIME = True
                break
            N_i = N_k + 1
            COCENTRIC = False
            while True:
                for x in range(N_i):
                    qqx = self.make_qq(N_i, x)
                    if self.cocentric(qqx):
                        COCENTRIC = True
                        self.break_from_EQ_by_vibration(qqx, N_i, degree=(np.random.ranf(1) - 0.5)/1000)
                if COCENTRIC: COCENTRIC = False
                else: break
            loss = self.update_single_sphere(N_i, ref_id=N_k)

        #X = chain_length -2
        #while X > 0:
        #    print('inspect the relation between {} and {}'.format(X - 1, X))
        #    qsr0 = self.inspect_QSR4syllogism(X - 1, X)
        #    print('the relation is ', qsr0)
        #    X = X - 1

        if self.orientation_fixed():
            loss = self.reverse_update(start=self.ent_num-1)

        if SHOW_DIAGRAM: self.show_sphere()
        #print(self.get_loss(0,1,grad=False))
        # update N_j to satisfy the relation to N_k while keeing the relation to N_i
        N_i, N_j, N_k = chain_length - 2, chain_length - 1, 0

        #print('construct4chain_n', data_i, chain_length)
        #qsr0 = self.inspect_QSR4syllogism(0,2)
        #print('qsr0', 0, 2, qsr0)

        loss = self.double_loss(N_j, N_k, N_i, grad=False)
        if self.centre_not_fixed() and loss > 0:
            #print('in construct4chain_n', N_j, N_k, N_i)
            loss = self.constraint_optimisation_of_sphere(data_i, opt_sphere=N_j, ref_sphere=N_k,
                                                          constraint_sphere=N_i, logger=logger)

        #print('after constraint_optimisation_of_sphere', data_i, chain_length)
        #qsr0 = self.inspect_QSR4syllogism(0, 2)
        #print('qsr0', qsr0)

        ## backward construction
        #count = 0
        if loss > 0 and chain_length > 3:
            qsr0 = self.inspect_QSR4syllogism(N_i, N_k)
            self.fix_one_and_optimise_two(N_k, N_i, fix_id = N_j)
            qsr1 = self.inspect_QSR4syllogism(N_i, N_k)
            if qsr0 == qsr1 and qsr0 == 'PO':
                #print('qsr0 == qsr1, check...')
                #print('qsr0', qsr0)
                #X = N_i
                #while X > 0:
                #    print('inspect the relation between {} and {}'.format(X-1, X))
                #    qsr0 = self.inspect_QSR4syllogism(X-1, X)
                #    print('the relation is ', qsr0)
                #    X = X -1

                loss = self.constraint_optimisation_of_sphere(data_i, opt_sphere=N_j, ref_sphere=N_k,
                                                              constraint_sphere=N_i, logger=logger)
                #count += 1
                #if self.out_of_time():
                #    self.OUT_OF_TIME = True
                #    break
            else:
                self.TB_0[N_i][N_k] = self.determine_syllogistic_relation(not_be=qsr0, can_be=qsr1)
                #break
        #print('construct4chain_n', chain_length, ' loss:', loss)
        return loss

    def constructive_learning4chain_algo(self, i, logger):
        # =======================================================================================
        # initial status is that all balls coincide
        # =======================================================================================
        # self.opt = optim.SGD(params=self.parameters(), lr=self.lr, weight_decay=self.w_decay)
        start_time_0 = time.time()
        self.task_timer = time.time() + self.task_duration
        self.train()
        g_loss = self.global_loss_chain()
        iter, loss3 = 0, np.inf
        refId, locId1, locId2 = 2, 1, 0
        if g_loss == 0:
            loss3 = g_loss
        else:
            while iter < self.epoch and loss3 > 0 and g_loss >0:
                for j in range(self.ent_num):
                    if self.ent_num - j < 3 or loss3 == 0: continue
                    loss3 = self.construct4chain_n(i, self.ent_num - j, logger)
                #if self.center_fixed == 2:
                #    loss = self.reverse_update(start=self.ent_num-1)
                if loss3 > 0:
                    if self.centre_not_fixed():  # not fixed
                        self.update_single_sphere(0, ref_id=2)  # backward
                        self.update_single_sphere(1, ref_id=2)  # backward
                        # loss = self.double_loss(1, 0, 2, grad=False)
                        loss3 = self.constraint_optimisation_of_sphere(i, opt_sphere=1, ref_sphere=0,
                                                                      constraint_sphere=2, logger=logger)
                    else:
                        self.update_single_sphere(locId1, ref_id=refId)
                        self.update_single_sphere(locId2, ref_id=refId)
                        self.update_single_sphere(locId2, ref_id=locId1)
                        refId, locId1, locId2 = (refId +1) % 3, (locId1+1) % 3, (locId2+1)% 3
                        self.update_two_spheres_fixing_ref_ori(locId1, locId2, ref_id=refId)
                        loss3 = self.get_triangle_loss(locId1, locId2, refId)
                iter += 1
        g_loss = self.global_loss_chain()
        if SHOW_DIAGRAM: self.show_sphere()
        if self.fw:
            self.fw.write(
                "data: {}\nraw_data: {}\nmapping: {}\n".format(self.inputs, self.raw_input, self.ent_dict))
            self.fw.write("\n total time CL: data_{}:  {}".format(i, time.time() - start_time_0))
            if g_loss == 0:
                self.fw.write("\nafter g_loss = {}\n".format(g_loss))
            else:
                self.fw.write("\nbackward loss of <0,1,2> = {}\n".format(loss3))
            self.save_spheres_to_file("the final configuration of backward phase only to check validity!")
        if self.out_of_time(): self.fw.write("\nconstruction stopped due to out of time\n")
        if g_loss == 0:
            return 0
        else:
            return loss3

    def global_loss_chain(self):
        g_loss = 0
        dLst = self.detach_list
        for N_k in range(self.ent_num-1):
            N_i = N_k + 1
            self.detach_list = [N_k]
            g_loss = g_loss + self.get_loss(N_i, N_k, grad=False, strict=True)
        g_loss += self.get_loss(0, self.ent_num-1, grad=False, strict=True)
        self.detach_list = dLst
        return g_loss

    def evaluate_constructed_diagrams(self, i, gloss = -1, flag='sat', logger=None):
        # Test
        logger.info("start testing data: " + str(i))
        predict_Deltalt = 'False'
        if gloss == np.inf: # out of time
            predict_Deltalt = 'UNKNOWN'
            if self.fw:
                self.fw.write('\n predict {}    raw_input {}\n'.format(predict_Deltalt,  self.raw_input))
        else:
            g_loss = gloss
            if g_loss == -1:
                with torch.no_grad():
                    self.eval()
                    g_loss = self.global_loss_chain()
            if flag == 'sat':
                if g_loss == 0:
                    predict_Deltalt = 'True'
            elif flag == 'valid':
                # negation is not satisfied
                if g_loss > 0:
                    predict_Deltalt = 'True'
            logger.info("g_loss = {}".format(g_loss))
            if self.fw:
                self.fw.write('\n predict {}  g_loss {} raw_input {}\n'.format(predict_Deltalt, g_loss, self.raw_input))

        logger.info("data_{} predict:  {}".format(i, predict_Deltalt))
        if self.fw:
            self.save_spheres_to_file(description="final configuration")

        return predict_Deltalt

    def forward(self, q, strict=False):
        tar_state = q[2]
        w = self.embeddings(q[0])
        v = self.embeddings(q[1])

        o_w, r_w = 0.9*torch.sin(w[:, :self.dim]), torch.sin(w[:, self.dim:])**2
        o_v, r_v = 0.9*torch.sin(v[:, :self.dim]), torch.sin(v[:, self.dim:])**2
        dis = torch.norm(o_w - o_v, p=self.p, dim=-1)

        cur_state,tar_state = self.get_state(o_w, r_w, o_v, r_v, target=tar_state, strict=strict)

        if self.orientation_fixed():
            l_w = torch.norm(o_w, p=self.p, dim=-1)
            alpha_w = o_w / l_w  # pre-trained vectors have the length 1
            alpha_w = alpha_w.detach()

            l_v = torch.norm(o_v, p=self.p, dim=-1)
            alpha_v = o_v / l_v  # pre-trained vectors have the length 1
            alpha_v = alpha_v.detach()

            return getattr(self, self.routes[tar_state][cur_state])(o_w=o_w, r_w=r_w, l_w=l_w, alpha_w=alpha_w,
                                                                    o_v=o_v, r_v=r_v, l_v=l_v, alpha_v=alpha_v,
                                                                    pole = True)
        elif q[0].item() in self.detach_list:
            #w = w.detach()
            o_w = o_w.detach()
            r_w = r_w.detach()
        elif q[1].item() in self.detach_list:
            #v = v.detach()
            o_v = o_v.detach()
            r_v = r_v.detach()

        return getattr(self, self.routes[tar_state][cur_state])(o_w=o_w, r_w=r_w, o_v=o_v, r_v=r_v, pole=False)

    def evaluate(self, q, special=False):
        tar_state = q[2]
        if special:
            return getattr(self, tar_state)(q)
        else:
            w = self.embeddings(q[0])
            v = self.embeddings(q[1])
            o_w, r_w = 0.9*torch.sin(w[:, :self.dim]), torch.sin(w[:, self.dim:])**2
            o_v, r_v = 0.9*torch.sin(v[:, :self.dim]), torch.sin(v[:, self.dim:])**2

            if len(tar_state) > 1:
                tar = tar_state[1:]
                if getattr(self, tar)(o_w, r_w, o_v, r_v):
                    return False
                else:
                    return True
            if getattr(self, tar_state)(o_w, r_w, o_v, r_v):
                return True
            else:
                return False

    def repair_broken_relation(self, id3, fixed_id, logger=None):
        self.detach_list = [fixed_id]
        qq = self.make_qq(id3, fixed_id)
        opt_fun = self.get_opt_fun(qq)
        loss, last_loss = 0, np.inf
        kappa = 2
        while opt_fun !="DONE":
            #print("reparing...opt_fun:", opt_fun)
            loss = self.forward(qq)

            self.opt4cop = optim.SGD(params=self.parameters(), lr=self.lr/kappa, weight_decay=self.w_decay)
            self.opt4cop.zero_grad()
            loss.backward()
            self.opt4cop.step()

            if last_loss == np.inf:
                last_loss = loss
            elif last_loss - loss < 10* self.zero:
                return loss
            #elif last_loss < loss:
            #    kappa = min(kappa * 2, 10)
            if loss.item() == 0:
                #self.center_fixed = center_fixed
                return 0
            last_loss = loss
            opt_fun = self.get_opt_fun(qq)
        return loss

    def random_satellites_around(self, id, ref_id=0):
        with torch.no_grad():
            weights = self.embeddings.weight.detach().cpu().numpy() #.tolist()
            sphere_id = weights[id][:-1],
            ref_point = weights[ref_id][:-1]
            dis = np.linalg.norm(sphere_id - ref_point)
            vecs = [np.random.ranf(self.dim) -0.5  for _ in range(self.rand_rotate_num)]
            points = [ref_point + dis *v/np.linalg.norm(v) for v in vecs]
            # vis_util.show_spheres(weights=[], id=id, centres=points)
            return points

    def choose_sphere_with_mininum_loss(self, candi_centres, id, ref_id = 0, his=False):
        min_loss = self.get_loss(id, ref_id, grad=False)
        with torch.no_grad():
            weights = self.embeddings.weight.detach().cpu().numpy()  # .tolist()
            # vis_util.show_spheres(weights=weights, id=id, centres=candi_centres)
            best_centres = copy.deepcopy(weights[id][:-1])
            for c in candi_centres:
                weights[id][:-1] = c
                self.embeddings.weight.copy_(torch.FloatTensor(weights))
                loss = self.get_loss(id, ref_id, grad=False)
                if min_loss > loss:
                    best_centres = c
                    min_loss = loss
            weights[id][:-1] = best_centres
            self.embeddings.weight.copy_(torch.FloatTensor(weights))

    def constraint_optimisation_with_random_rotation(self, i, id3, fixed_id2, fixed_id, logger=None):
        """
        randomly choose self.rand_rotate_num for sphere id3 around sphere fixed_id, choose the location having the
        minimal loss with the sphere fixed_id2
        :param i:
        :param id3:
        :param fixed_id2:
        :param fixed_id:
        :param logger:
        :return:
        """
        logger.info("data_{}: {} constraint optimisation with random rotation around {}".format(i, id3, fixed_id))
        #print('in constraint_optimisation_with_random_rotation',  id3, fixed_id2, fixed_id)
        d_loss = np.inf
        last_state = ''
        # steps = [1, 2]
        self.his_states = defaultdict()
        self.his_config = defaultdict()
        self.flag_loss = defaultdict()
        self.detach_list = [fixed_id2, fixed_id]
        FirstRound = True
        while d_loss > 0:
            # print('in random rotation..')
            if self.out_of_time():
                self.OUT_OF_TIME = True
                break
            if FirstRound: # center not fixed
                candi_centres = self.random_satellites_around(id3, ref_id=fixed_id)
                self.choose_sphere_with_mininum_loss(candi_centres, id3, ref_id=fixed_id2, his=False)
                FirstRound = False
            #d_loss_0 = self.double_loss(id3, fixed_id, fixed_id2, grad=False)
            #print("before cop, spheres {}, {}, {}  {} d_loss_0 {}".format(id3, fixed_id2, fixed_id,
            #                                                              self.embeddings.weight, d_loss_0))
            self.opt.zero_grad()
            d_loss0 = self.double_loss(id3, fixed_id, fixed_id2)
            if d_loss0 == 0: break
            d_loss0.backward()
            self.opt.step()
            d_loss00 = self.double_loss(id3, fixed_id, fixed_id2)
            loss = self.repair_broken_relation(id3, fixed_id, logger=logger)
            d_loss_1 = self.double_loss(id3, fixed_id2, fixed_id, grad=False)
            if d_loss0 - d_loss_1 > 0:
                if d_loss0 - d_loss_1 < 100 * self.zero:
                    self.opt = self.opt100
                elif d_loss_1 > 10:
                    self.opt = self.optBig
                else:
                    self.opt = self.optInit

            target2 = self.get_target_rel(id3, fixed_id2)
            qq32 = self.make_qq(id3, fixed_id2)
            current_state, tar_state = self.get_state(qq32, target=target2)

            if current_state not in self.his_states.keys():  # compare with the best loss in the history
                self.his_states[current_state] = d_loss_1
                self.flag_loss[current_state] = 0
            elif self.his_states[current_state] - d_loss_1 < 10*self.zero:
                self.flag_loss[current_state] += 1
                if self.flag_loss[current_state] > self.NUM_OF_ATTEMPTS: break
            else:
                self.his_states[current_state] = d_loss_1

        return self.double_loss(id3, fixed_id2, fixed_id, grad=False)

    def get_state(self, o_w, r_w=None, o_v=None, r_v=None, target=None, strict = False):
        if o_w is not None and r_w is None and o_v is None and r_v is None:
            q = o_w
            if q[1] in self.detach_list:
                w = self.embeddings(q[0])
                v = self.embeddings(q[1])
                o_w, r_w = 0.9*torch.sin(w[:, :self.dim]), torch.sin(w[:, self.dim:])**2
                o_v, r_v = 0.9*torch.sin(v[:, :self.dim]), torch.sin(v[:, self.dim:])**2
                target = target
            else:
                w = self.embeddings(q[1])
                v = self.embeddings(q[0])
                o_w, r_w = 0.9*torch.sin(w[:, :self.dim]), torch.sin(w[:, self.dim:])**2
                o_v, r_v = 0.9*torch.sin(v[:, :self.dim]), torch.sin(v[:, self.dim:])**2
                target = inverse_relation(target)

        if target == "D":
            if self.D(o_w, r_w, o_v, r_v, strict=strict):
                return "D", target
            elif self.EQ(o_w, r_w, o_v, r_v):
                return "EQ", target
            elif self.PP(o_w, r_w, o_v, r_v):
                return "PP", target
            elif self.PPbar(o_w, r_w, o_v, r_v):
                return "PPbar", target
            elif self.PO1(o_w, r_w, o_v, r_v):
                return "PO1", target
            elif self.PO2(o_w, r_w, o_v, r_v):
                return "PO2", target
            else:
                print('target=', target )
                raise NotImplementedError("get_state ???")
        elif target == "-D":
            if self.D(o_w, r_w, o_v, r_v):
                return "D", target
            elif self.ND(o_w, r_w, o_v, r_v):
                return "ND", target
            else:
                raise NotImplementedError("get_state ???")
        elif target == "P":
            if self.P(o_w, r_w, o_v, r_v, strict=strict):
                return "P", target
            elif self.D(o_w, r_w, o_v, r_v, strict=strict):
                return "D", target
            #elif self.qEQ(o_w, r_w, o_v, r_v):
            #    return "qEQ", target
            elif self.PO1(o_w, r_w, o_v, r_v):
                return "PO1", target
            elif self.PO2(o_w, r_w, o_v, r_v):
                return "PO2", target
            elif self.PPbar(o_w, r_w, o_v, r_v):
                return "PPbar", target
            else:
                raise NotImplementedError("get_state ???")
        elif target == "-P":
            if self.NP(o_w, r_w, o_v, r_v, strict=strict):
                return "NP", target
            elif self.P(o_w, r_w, o_v, r_v, strict=strict):
                return "P", target
            else:
                raise NotImplementedError("get_state ???")
        elif target == "Pbar":
            if self.Pbar(o_w, r_w, o_v, r_v, strict=strict):
                return "Pbar", target
            elif self.D(o_w, r_w, o_v, r_v, strict=strict):
                return "D", target
            #elif self.qEQ(o_w, r_w, o_v, r_v):
            #    return "qEQ", target
            elif self.PO3(o_w, r_w, o_v, r_v):
                return "PO3", target
            elif self.PO4(o_w, r_w, o_v, r_v):
                return "PO4", target
            elif self.PP(o_w, r_w, o_v, r_v):
                return "PP", target
            else:
                raise NotImplementedError("get_state ???")
        elif target == "-Pbar":
            if self.NPbar(o_w, r_w, o_v, r_v, strict=strict):
                return 'NPbar', target
            elif self.Pbar(o_w, r_w, o_v, r_v, strict=strict):
                return "Pbar", target
            else:
                raise NotImplementedError("get_state ???")
        elif target == "PO":
            if self.PO(o_w, r_w, o_v, r_v):
                return 'PO', target
            elif self.D(o_w, r_w, o_v, r_v):
                return "D", target
            elif self.P(o_w, r_w, o_v, r_v):
                return "P", target
            elif self.Pbar(o_w, r_w, o_v, r_v):
                return "Pbar", target
            else:
                raise NotImplementedError("get_state ???")
        else:
            print('target=', target, q)
            raise NotImplementedError("get_state ???")

    def cocentric(self, q):
        w = self.embeddings(q[0])  # k
        v = self.embeddings(q[1])  # j
        o_w, r_w = 0.9*torch.sin(w[:, :self.dim]), torch.sin(w[:, self.dim:])**2
        o_v, r_v = 0.9*torch.sin(v[:, :self.dim]), torch.sin(v[:, self.dim:])**2
        if torch.norm(o_w - o_v, p=self.p, dim=-1) ==0:
            return True
        else:
            return False

    def break_from_EQ_by_vibration(self, q, vib_id, degree=1):
        w = self.embeddings(q[0])  # k
        v = self.embeddings(q[1])  # j
        o_w, r_w = 0.9*torch.sin(w[:, :self.dim]), torch.sin(w[:, self.dim:])**2
        o_v, r_v = 0.9*torch.sin(v[:, :self.dim]), torch.sin(v[:, self.dim:])**2
        with torch.no_grad():
            weights = self.embeddings.weight.detach().cpu().numpy().tolist()
        o_w = o_w.detach().cpu().numpy()
        o_v = o_v.detach().cpu().numpy()
        while True:
            if (np.linalg.norm(o_w - o_v) > 0.0):
                break
            o_shift = torch.rand((1, self.dim))/100
            l = np.linalg.norm(o_shift)
            o_w = o_w + degree*o_shift.numpy()/l
        weights[vib_id][:-1] = o_w[0]
        # weights[vib_id][-1] = weights[vib_id][-1] + self.eps
        with torch.no_grad():
            self.embeddings.weight.copy_(torch.FloatTensor(weights))


    def get_opt_fun(self, q, strict=False):
        """
        if the two balls are coincide, flucturate one of the central point
        :param qq:
        :return:
        """
        with torch.no_grad():
            tar_state = q[2]
            cur_state, tar_state = self.get_state(q, target=tar_state, strict=strict)
            op_fun = self.routes[tar_state][cur_state]
            return op_fun

    def double_loss(self, i, j, k, grad=True):
        self.detach_list = [j, k]
        return self.get_loss(i, j, grad=grad) + self.get_loss(i, k, grad=grad)

    def get_loss(self, i, j, grad=False, strict=True):
        qq = self.make_qq(i, j)
        opt_fun = self.get_opt_fun(qq, strict=strict)
        # print('opt_fun', opt_fun)
        if opt_fun != "DONE":
            loss = self.forward(qq, strict=strict)
            if loss == 0: return 0
            if not grad: loss = loss.item()
        else: loss = 0
        return loss

    def loss_fixing_ori3(self, N_k, N_i, ref_id):
        def pair_loss(x, y):
            loss_ki = 0
            if self.TB_0[x][y] != '':
                target1 = self.TB_0[x][y]
            else:
                target1 = inverse_relation(self.TB_0[x][y])

            qq_ki = [torch.LongTensor([x]).to(self.device), torch.LongTensor([y]).to(self.device), target1]
            if self.get_opt_fun(qq_ki) != "DONE":
                loss_ki = self.forward(qq_ki)
            return loss_ki
        return pair_loss(N_k, N_i) + pair_loss(N_i, ref_id) + pair_loss(N_k, ref_id)

    def save_spheres_to_file(self, description="status of configuration"):
        blst = []
        for j in range(self.ent_num):
            q = torch.LongTensor([j]).cuda() if next(self.parameters()).is_cuda else torch.LongTensor([j])
            w = self.embeddings(q)
            o_w, r_w = 0.9*torch.sin(w[:, :self.dim]), torch.sin(w[:, self.dim:])**2
            blst.append(' '.join(['('] + [','.join([str(s) for s in o_w.cpu().detach().numpy().tolist()[0] if len(str(s).strip())>0])] + ['), '] +
                                 [str(s) for s in r_w.cpu().detach().numpy().tolist()[0]]))
        spheres = ', '.join(blst)
        self.fw.write('\n{} {}\n'.format(description, spheres))
        self.fw.flush()

    #
    # Spatial predicates
    #

    def poincare_distance(self, o_w, o_v):
        lambda_w = 2/(1 - torch.norm(o_w, p=self.p, dim=-1)*torch.norm(o_w, p=self.p, dim=-1))
        lambda_v = 2 / (1 - torch.norm(o_v, p=self.p, dim=-1)*torch.norm(o_v, p=self.p, dim=-1))
        value = 1 + torch.norm(o_w - o_v, p=self.p, dim=-1)*torch.norm(o_w - o_v, p=self.p, dim=-1) * lambda_w * lambda_v /2
        # print("p_dis:", torch.acosh(value))
        return torch.acosh(value)

    def poincar_centre(self, o, r):
        l = torch.norm(o, p=self.p, dim=-1)
        poincareL = torch.atanh(r+l) + torch.atanh(l-r)
        return torch.tanh(poincareL/2) * o / l

    def poincare_radius(self, r, o=0):
        # print("p_radius: ", 2*torch.atanh(r))
        l = torch.norm(o, p=self.p, dim=-1)
        return torch.atanh(r+l) - torch.atanh(l-r)
        #return r


    def D(self, o_w, r_w, o_v, r_v, strict = True):
        """
        :param o_w: Euclidean points of O_w
        :param r_w: Euclidean radius of O_w
        :param o_v: Euclidean points of O_v
        :param r_v: Euclidean radius of O_v
        :param strict:
        :return:
        """
        if strict: gap = 0
        else:gap = 0# self.zero
        o_w_p = self.poincar_centre(o_w, r_w)
        o_v_p = self.poincar_centre(o_v, r_v)
        return (self.poincare_distance(o_w_p, o_v_p) - (self.poincare_radius(r_w, o=o_w) + self.poincare_radius(r_v, o=o_v)) >= gap).item()

    def ND(self, o_w, r_w, o_v, r_v, strict = False):
        return not self.D(o_w, r_w, o_v, r_v, strict=strict)

    def PO(self, o_w, r_w, o_v, r_v, strict = False):
        if strict: gap = 0
        else:gap = 0# self.zero
        o_w_p = self.poincar_centre(o_w, r_w)
        o_v_p = self.poincar_centre(o_v, r_v)
        return (torch.abs(self.poincare_radius(r_w, o=o_w) - self.poincare_radius(r_v, o=o_v)) < self.poincare_distance(o_w_p, o_v_p) + gap and
                    self.poincare_distance(o_w_p, o_v_p) - gap < (self.poincare_radius(r_w, o=o_w) + self.poincare_radius(r_v, o=o_v))).item()

    def PO1(self, o_w, r_w, o_v, r_v):
        o_w_p = self.poincar_centre(o_w, r_w)
        o_v_p = self.poincar_centre(o_v, r_v)
        return self.PO(o_w, r_w, o_v, r_v) and (self.poincare_distance(o_w_p, o_v_p) > self.poincare_radius(r_v, o=o_v)).item()


    def PO2(self, o_w, r_w, o_v, r_v):
        o_w_p = self.poincar_centre(o_w, r_w)
        o_v_p = self.poincar_centre(o_v, r_v)
        return self.PO(o_w, r_w, o_v, r_v) and (self.poincare_distance(o_w_p, o_v_p) <= self.poincare_radius(r_v, o=o_v)).item()

    def PO3(self, o_w, r_w, o_v, r_v):
        return self.PO(o_w, r_w, o_v, r_v) and (self.poincare_radius(r_w, o=o_w) < self.poincare_radius(r_v, o=o_v))

    def PO4(self, o_w, r_w, o_v, r_v):
        return self.PO(o_w, r_w, o_v, r_v) and (self.poincare_radius(r_w, o=o_w) >= self.poincare_radius(r_v, o=o_v))

    def P(self, o_w, r_w, o_v, r_v, strict = False):
        if strict: gap = 0
        else: gap = 0# self.zero
        o_w_p = self.poincar_centre(o_w, r_w)
        o_v_p = self.poincar_centre(o_v, r_v)
        return (self.poincare_distance(o_w_p, o_v_p) + self.poincare_radius(r_w, o=o_w) + gap <= self.poincare_radius(r_v, o=o_v))

    def EQ(self, o_w, r_w, o_v, r_v):
        return self.P(o_v, r_v, o_w, r_w) and self.Pbar(o_v, r_v, o_w, r_w)

    def qEQ(self, o_w, r_w, o_v, r_v):
        gap = self.zero * 1000000
        o_w_p = self.poincar_centre(o_w, r_w)
        o_v_p = self.poincar_centre(o_v, r_v)
        dis = self.poincare_distance(o_w_p, o_v_p).item()
        if self.poincare_radius(r_w, o=o_w) >= self.poincare_radius(r_v, o=o_v):
            return (self.poincare_radius(r_w, o=o_w) - self.poincare_radius(r_v, o=o_v) < gap) and dis < gap
        else:
            return (self.poincare_radius(r_v, o=o_v) - self.poincare_radius(r_w, o=o_w) < gap) and dis < gap

    def NP(self, o_w, r_w, o_v, r_v, strict = False):
        return not self.P(o_w, r_w, o_v, r_v, strict=strict)

    def PP(self, o_w, r_w, o_v, r_v):
        return self.P(o_w, r_w, o_v, r_v) and not self.EQ(o_w, r_w, o_v, r_v)

    def Pbar(self, o_w, r_w, o_v, r_v, strict = False):
        return self.P(o_v, r_v, o_w, r_w, strict=strict)

    def NPbar(self, o_w, r_w, o_v, r_v, strict=False):
        return not self.Pbar(o_w, r_w, o_v, r_v, strict=strict)

    def PPbar(self, o_w, r_w, o_v, r_v):
        return self.Pbar(o_w, r_w, o_v, r_v) and not self.EQ(o_w, r_w, o_v, r_v)

    #
    #
    def DONE(self, o_w=0, r_w=1, o_v=0, r_v=1, pole=False, alpha_w=0, l_w=1, alpha_v=0, l_v=1):
        return 0

    def INC_DIS(self,o_w=0, r_w=1, o_v=0, r_v=1,  pole=False, alpha_w = 0, l_w=1, alpha_v = 0, l_v=1):
        o_w_p = self.poincar_centre(o_w, r_w)
        o_v_p = self.poincar_centre(o_v, r_v)
        if pole:
            dist = F.relu((self.eps+ r_v + r_w.item() - torch.norm(alpha_w*l_w - alpha_v*l_v, p=self.p, dim=-1))[0])
        else:
            dist = F.relu((self.eps+ self.poincare_radius(r_v, o=o_v) + self.poincare_radius(r_w, o=o_w).item() - self.poincare_distance(o_w_p, o_v_p)[0]))
        return dist

    def DEC_DIS(self,o_w=0, r_w=1, o_v=0, r_v=1,  pole=False, alpha_w = 0, l_w=1, alpha_v = 0, l_v=1):
        o_w_p = self.poincar_centre(o_w, r_w)
        o_v_p = self.poincar_centre(o_v, r_v)
        if pole:
            dist = F.relu((torch.norm(alpha_v*l_v - alpha_w*l_w, p=self.p, dim=-1))[0])
        else:
            dist = F.relu((self.poincare_distance(o_w_p, o_v_p))[0])
        return dist


    def INC_R(self,o_w=0, r_w=1, o_v=0, r_v=1,  pole=False, alpha_w = 0, l_w=1, alpha_v = 0, l_v=1):
        dist = F.relu((self.eps+ self.poincare_radius(r_v, o=o_v) - self.poincare_radius(r_w, o=o_w)[0]))
        return dist

    def DEC_R(self,o_w=0, r_w=1, o_v=0, r_v=1,  pole=False, alpha_w = 0, l_w=1, alpha_v = 0, l_v=1):
        dist = F.relu(self.poincare_radius(r_w, o=o_w)[0]+self.eps)
        return dist

    def INC_DIS_DEC_R(self,o_w=0, r_w=1, o_v=0, r_v=1,  pole=False, alpha_w = 0, l_w=1, alpha_v = 0, l_v=1):
        o_w_p = self.poincar_centre(o_w, r_w)
        o_v_p = self.poincar_centre(o_v, r_v)
        if pole:
            dist = F.relu((self.eps+ r_v - torch.norm(alpha_w*l_w - alpha_v*l_v, p=self.p, dim=-1)+ r_w)[0])
        else:
            dist = F.relu((self.eps+ self.poincare_radius(r_v, o=o_v) - self.poincare_distance(o_w_p, o_v_p) + self.poincare_radius(r_w, o=o_w))[0])
        return dist

    def INC_DIS_INC_R(self,o_w=0, r_w=1, o_v=0, r_v=1,  pole=False, alpha_w = 0, l_w=1, alpha_v = 0, l_v=1):
        o_w_p = self.poincar_centre(o_w, r_w)
        o_v_p = self.poincar_centre(o_v, r_v)
        if pole:
            dist = F.relu((self.eps+ r_v - torch.norm(alpha_w*l_w - alpha_v*l_v, p=self.p, dim=-1)-r_w)[0])
        else:
            dist = F.relu((self.eps+ self.poincare_radius(r_v, o=o_v) - self.poincare_distance(o_w_p, o_v_p)- self.poincare_radius(r_w, o=o_w))[0])
        return dist

    def DEC_DIS_DEC_R(self,o_w=0, r_w=1, o_v=0, r_v=1,  pole=False, alpha_w = 0, l_w=1, alpha_v = 0, l_v=1):
        o_w_p = self.poincar_centre(o_w, r_w)
        o_v_p = self.poincar_centre(o_v, r_v)
        if pole:
            dist = F.relu((torch.norm(alpha_v*l_v - alpha_w*l_w, p=self.p, dim=-1) + r_w)[0])
        else:
            dist = F.relu((self.poincare_distance(o_w_p, o_v_p) + self.poincare_radius(r_w, o=o_w))[0])
        return dist

    def DEC_DIS_INC_R(self,o_w=0, r_w=1, o_v=0, r_v=1,  pole=False, alpha_w = 0, l_w=1, alpha_v = 0, l_v=1):
        o_w_p = self.poincar_centre(o_w, r_w)
        o_v_p = self.poincar_centre(o_v, r_v)
        if pole:
            dist = F.relu(self.eps+ self.poincare_radius(r_v, o=o_v)+ (torch.norm(alpha_v*l_v - alpha_w*l_w, p=self.p, dim=-1)  - r_w)[0])
        else:
            dist = F.relu(self.eps+ self.poincare_radius(r_v, o=o_v)+ (self.poincare_distance(o_w_p, o_v_p) - self.poincare_radius(r_w, o=o_w))[0])
        return dist


