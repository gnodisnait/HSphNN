import copy
import json
import openai
import logging
from torch.utils.tensorboard import SummaryWriter

from load_data import *
from models_3 import *
from eval_exp2 import *

TIME_BREAK=15
NOT_IN_TEST=True


class GPT_SphNN_dualMind:
    def __init__(self, api_key, vgpt, rolle):
        openai.api_key = api_key
        self.dialog = [{"role":"system", "content":rolle}]
        self.task = None
        self.ndic = {'INSIDE':'P', 'OUTSIDE':'D', 'DISCONNECTS':'D','DISCONNECT':'D', 'CONNECTED':'C', 'DISCONNECTING':'D', 'OVERLAPS':'PO',
                     'OVERLAP':'PO', 'OVERLAPPING':'PO', 'CONTAINS':'Pbar', 'CONTAINING':'Pbar', 'CONTAIN':'Pbar', 'NOT':'not',
                     'SOME':'-D', 'SOME-NOT':'-P', 'ALL':'P', 'NO':'D'}
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        self.model_checker = None
        self.vgpt = vgpt
        self.sphnn_decision = ""
        self.chatgpt_decision = ""
        self.chatgpt_has_explanation = False
        self.chatgpt_last_explanation = []
        self.feedback = dict()
        self.promp0 = ""
        self.initialize_model_checker()
        self.fw = None
        self.log_json = {"id": 0,
                    "task":"",
                    "sphnn_d": "",
                    "sphnn_feedback_num": 0}
        self.chatgpt_attempt = {
                        "sphnn_feedback": "",
                        "chatgpt_raw_answer": "",
                        "chatgpt_answer": "",
                        "chatgpt_d": "",
                        "chatgpt_expl": "",
                        "repeat_last_answer": 0,
                        "d_type": "",
                        "expl_type": "", #
                    }

    def clear_log_json(self):
        self.log_json = {"id": 0,
                         "task": "",
                         "sphnn_d": "",
                         "sphnn_feedback_num": 0}
        self.feedback = dict()
        self.sphnn_decision = "0"
        self.clear_chatgpt_attempt()

    def clear_chatgpt_attempt(self):
        self.chatgpt_attempt = {
            "sphnn_feedback": "",
            "chatgpt_raw_answer": "",
            "chatgpt_answer": "",
            "chatgpt_d": "",
            "repeat_last_answer": 0,
            "chatgpt_expl": "",
            "d_type": "",
            "expl_type": "",  #
        }
        self.chatgpt_decision = "0"
        self.model_checker.TB_0 = []

    def get_completion(self, prompt, max_num=3):
        fine_format = False
        count = 0
        while not fine_format and count < max_num:
            messages = [{"role": "user", "content": prompt}]
            time0 = time.time()
            response = openai.ChatCompletion.create(
                model=self.vgpt,
                messages=messages,
                temperature=0,)  # this is the degree of randomness of the model's output
            answer = response.choices[0].message["content"]
            self.chatgpt_attempt["chatgpt_raw_answer"] = answer
            duration = time.time() - time0
            count += 1
            print("answer:", answer)

            if self.have_negative_relation(copy.deepcopy(answer)) or '(' not in answer:
                self.get_chatgpt_answer(answer)
                if self.chatgpt_decision == "UNSAT":
                    return answer
                feedback = f"""\nUse ''('''' and ''')''' and '''inside''' and '''disconnect''' and '''overlap''' in your explaination, e.g. (circle X, inside, circle Y)."""
                prompt = self.prompt0 + feedback
                time.sleep(TIME_BREAK)
            else:
                fine_format = True

        self.get_chatgpt_answer(answer)
        return answer

    def initialize_model_checker(self):
        with open('params.json', 'r') as ijh:
            params = json.load(ijh)
            log_dir = params['exp2']["output_dir"] + "%s_%s" % (params['exp2']["data_dir"].split("/")[-2], params['exp2']["log"])
            writer = SummaryWriter(log_dir)
            device = torch.device("cuda" if params['exp2']["cuda"] and torch.cuda.is_available() else "cpu")

            torch.backends.cudnn.deterministic = True
            random.seed(params['exp2']["seed"])
            np.random.seed(params['exp2']["seed"])
            torch.manual_seed(params['exp2']["seed"])
            if params['exp2']["cuda"] and torch.cuda.is_available():
                torch.cuda.manual_seed(params['exp2']["seed"])

            model = HyperbolicSphNN(params['exp2']["cuda"], params['exp2']["lr"], params['exp2']["w_decay"], params['exp2']["save"],
                          None, False, timer=params['exp2']["timer"], save_dir=params['exp2']["output_dir"],
                          pre_train=params['exp2']["pre_train"], dim=params['exp2']["dim"],
                          eps=params['exp2']["eps"], epoch=params['exp2']["epoch"],
                          cop=params['exp2']["cop"], rand_rotate_num=params['exp2']["rand_rotate_num"],
                          center_fixed=params['exp2']["center_fixed"], p=params['exp2']["p"], init_loc=params['exp2']["init_loc"],
                          init_r=params['exp2']["init_r"], pre_train_dic={}).to(device)
            opt = optim.SGD(params=model.parameters(), lr=params['exp2']["lr"], weight_decay=params['exp2']["w_decay"])
            optBig = optim.SGD(params=model.parameters(), lr=params['exp2']["lr"] * 1000, weight_decay=params['exp2']["w_decay"])
            opt100 = optim.SGD(params=model.parameters(), lr=params['exp2']["lr"] * 100, weight_decay=params['exp2']["w_decay"])
            optInit = optim.SGD(params=model.parameters(), lr=params['exp2']["lr"], weight_decay=params['exp2']["w_decay"])

            self.logger.info(model)
            self.logger.info(opt)

            model.opt = opt
            model.optInit = opt
            model.opt100 = opt100
            model.optBig = optBig
            self.model_checker = model

    def initialise_data(self, entdic):
        self.model_checker.ent_num = len(entdic)
        self.model_checker.inputs = self.task["data"]
        self.model_checker.raw_input = self.task["raw"]
        self.model_checker.entdic = entdic

    def add_to_log(self, content, key = ''):
        self.fw.write("\n{} {}\n".format(key, content))
        self.fw.flush()

    def get_chatgpt_answer(self, answer):
        if "cannot" in answer[:20].lower() or "incompatible" in answer[:100].lower():
            self.chatgpt_decision = "UNSAT"
        else:
            self.chatgpt_decision = "SAT"
        self.chatgpt_attempt["chatgpt_d"] = self.chatgpt_decision

    def formalize_data(self, answer):
        def switch_12element(lst):
            return [lst[0], lst[2], lst[1], lst[3]]

        def take_first_parenthesis(answer):
            sIndex = answer.index('(')
            eIndex = answer.index(')')
            statement, answer1 = answer[sIndex+1:eIndex], answer[eIndex+1:]
            return statement.replace('circle',''), answer1

        lst, ndata = [], []
        while '(' in answer:
            s, answer = take_first_parenthesis(answer)
            elst = [x for x in [e.strip() for e in re.split(',| ', s)] if len(x) > 0]
            ndata.append([self.ndic.get(x.upper(), '') for x in elst if len(str(self.ndic.get(x.upper(), ''))) > 0]  + [None])
            lst.append(s)
        ndata = [switch_12element(l) for l in ndata if len(l) == 4]
        return ndata[:3]

    def have_negative_relation(self, answer):
        def take_first_parenthesis(answer):
            sIndex = answer.index('(')
            eIndex = answer.index(')')
            statement, answer1 = answer[sIndex+1:eIndex], answer[eIndex+1:]
            return statement.replace('circle',''), answer1

        while '(' in answer:
            s, answer = take_first_parenthesis(answer)
            if "not" in s or "complement" in s: return True
        return False

    def chatgpt_give_an_early_answer(self, explanation):
        for i in range(self.log_json["sphnn_feedback_num"]):
            if self.chatgpt_decision == self.log_json[i]["chatgpt_d"]  and explanation == self.log_json[i]["chatgpt_expl"]:
                return True, i
        return False, ""

    def get_answer_of_task_by_model_construction(self, M=2):
        # step 1: SphNN first check whether it is satifiable.
        self.model_checker.set_criteria(self.model_checker.ent_num,
                                        self.model_checker.inputs[1],
                                        self.model_checker.ent_dict, self.logger)
        self.model_checker.TB_origin = copy.deepcopy(self.model_checker.TB_0)

        Mcount = 1
        print("SphNN is solving the task itself...")
        self.model_checker.re_initialise_spheres()
        loss = self.model_checker.constructive_learning4chain_algo(self.task["num"], self.logger)
        while Mcount < M and loss > 0:
            self.model_checker.re_initialise_spheres()
            loss = self.model_checker.constructive_learning4chain_algo(self.task["num"], self.logger)
            if loss == 0:
                pprint("loss reaches 0 not in first epoch!!!!")
                pprint(self.task)
            Mcount += 1
        if loss == 0:
            self.sphnn_decision = "SAT"
        else:
            self.sphnn_decision = "UNSAT"
        self.log_json["sphnn_d"] = self.sphnn_decision
        return self.sphnn_decision

    def check_explanation_by_model_construction(self, explanation):
        """
        # step 2: SphNN check the explanation of the answer.
        """
        explanation_data = self.formalize_data(explanation)
        self.chatgpt_attempt["chatgpt_expl"] = explanation_data

        if self.log_json["sphnn_feedback_num"] > 0:
            # last_id = str(self.log_json["sphnn_feedback_num"] -1)
            flag, _ = self.chatgpt_give_an_early_answer(explanation_data)
            if flag: return "SAME_ANSWER_In_Past"

        Not_HLU0 = len(explanation_data) == 3 and explanation_data[0] != explanation_data[1] and   \
                    explanation_data[0] != explanation_data[2] and \
                    explanation_data[2] != explanation_data[1]
        if not Not_HLU0:
            #self.add_to_log(explanation_data, key="HLU0 with partial information")
            self.chatgpt_attempt["expl_type"] = "HLU0"
            return "HLU0"

        HLU1, self.chatgpt_has_explanation = self.model_checker.acquire_entity_relations(explanation_data, self.logger)
        "consistent with the relations in the task, has three relations"
        if HLU1:
            #self.add_to_log(explanation_data, key="HLU1 with irrelevant relations")
            self.chatgpt_attempt["expl_type"] = "HLU1"
            return "HLU1"
        elif not self.chatgpt_has_explanation:
            #self.add_to_log(explanation_data, key="HLU0 with partial information")
            self.chatgpt_attempt["expl_type"] = "HLU0"
            return "HLU0"

        print("SphNN is checking the explanation...")
        #if self.chatgpt_last_explanation and self.chatgpt_give_the_same_explanation():
        #    return "SAME_ERROR_EXPL"

        loss = self.model_checker.constructive_learning4chain_algo(self.task["num"], self.logger)

        if loss > 0 and self.sphnn_decision == "UNSAT":
            #self.add_to_log(explanation_data, key='   correct explanation')
            self.chatgpt_attempt["expl_type"] = "expl"
            return "expl"

        elif loss == 0 and self.sphnn_decision == "UNSAT":
            #self.add_to_log(explanation_data, key='  HLU2')
            self.chatgpt_attempt["expl_type"] = "HLU2"
            return "HLU2"

        elif loss == 0 and self.sphnn_decision == "SAT":
            #self.add_to_log(explanation_data, key='  correct explanation')
            self.chatgpt_attempt["expl_type"] = "expl"
            return "expl"

        elif loss > 0 and self.sphnn_decision == "SAT":
            #self.add_to_log(explanation_data, key='  HLU2')
            self.chatgpt_attempt["expl_type"] = "HLU2"
            return "HLU2"

    def prompt_deduction_with_model_checker(self, prompt, max_num = 10, Mepoch = 2, randSym = False):
        '''
        sphNN helps ChatGPT to solve SAT problems.
        :param sphNN: 
        :param ques: 
        :return:  'all qtpDVYAanNeU CNQuJJXxBjHR' 'all CNQuJJXxBjHR TpnFmxug5Ush'
        '''

        if NOT_IN_TEST:
            answer = self.get_completion(prompt)
            self.chatgpt_attempt["chatgpt_answer"] = answer
        else:
            # answer = "cannot"
            answer = "yes. (circle M0, inside, circle P), (circle P, inside, circle S), (circle S, overlaps, circle M0)"

        #print("first round answer:", answer)

        self.get_answer_of_task_by_model_construction(M=2)
        count = 0

        while count <= max_num:
            check_result = self.check_explanation_by_model_construction(answer)
            "expl | HLU0 | HLU1 |HLU2 | SAME_ANSWER_AS_LAST_TIME"
            self.chatgpt_attempt["d_type"] = self.chatgpt_decision == self.sphnn_decision
            if check_result == "SAME_ANSWER_In_Past":
                explanation_data = self.chatgpt_attempt["chatgpt_expl"]
                _, i = self.chatgpt_give_an_early_answer(explanation_data)

                self.log_json[count] = copy.deepcopy(self.log_json[i])
                self.chatgpt_attempt["repeat_last_answer"] += 1
                self.feedback[count] = f""" You had the same answer in the past:"""+ self.chatgpt_attempt["chatgpt_raw_answer"]
                self.feedback[count] += f""" You repeated the same error!"""
                if count == max_num:
                    with open(self.fw, 'w', encoding='utf8') as f:
                        json.dump(self.log_json, f)
                    pprint(self.log_json)
                    self.clear_log_json()
                    count += 1
                else:
                    count += 1
                    self.log_json["sphnn_feedback_num"] = count
                    time.sleep(TIME_BREAK)
                    if NOT_IN_TEST:
                        answer = self.get_completion(prompt)
                    else:
                        # answer = "cannot"
                        answer = "yes. (circle M0, inside, circle P), (circle P, inside, circle S), (circle S, overlaps, circle M0)"
            elif check_result == "expl" and self.chatgpt_decision == self.sphnn_decision:
                self.chatgpt_attempt["expl_type"] = "expl"
                self.log_json[count] = self.chatgpt_attempt
                #self.add_to_log(answer, key='ChatGPT makes correct decision. SphNN decides the task as {} and approves ChatGPT\'s answer after {} time of feedback.'.format(self.sphnn_decision, count))
                with open(self.fw, 'w', encoding='utf8') as f:
                    json.dump(self.log_json, f)
                    pprint(self.log_json)
                self.clear_log_json()
                break
            elif check_result == "HLU0" and self.chatgpt_decision == "UNSAT" and self.sphnn_decision == "UNSAT":
                self.chatgpt_attempt["expl_type"] = "expl"
                self.log_json[count] = self.chatgpt_attempt
                #self.add_to_log(answer, key='ChatGPT makes correct decision. SphNN decides the task as {} and approves ChatGPT\'s answer after {} time of feedback.'.format(self.sphnn_decision, count))
                with open(self.fw, 'w', encoding='utf8') as f:
                    json.dump(self.log_json, f)
                pprint(self.log_json)
                self.clear_log_json()
                break
            elif count == max_num:
                if check_result == "HLU0":
                    comment = f"""\n\n Explanation used partial information:!"""
                    comment +='ChatGPT decides as {}. SphNN decides the task as {} and refutes ChatGPT\'s explanation after {} time of feedback.'.format(self.chatgpt_decision,
                                                                                                                                                         self.sphnn_decision, count)
                elif check_result == "HLU1":
                    comment = f"""\n\n  Explanation used irrelevant information!"""
                    comment += 'ChatGPT decides as {}. SphNN decides the task as {} and refutes ChatGPT\'s decision after {} time of feedback.'.format(self.chatgpt_decision,
                                                                                                                                                        self.sphnn_decision, count)
                elif check_result == "HLU2":
                    comment = f"""\n\n  Explanation contains reasoning flaws!"""
                    comment += 'ChatGPT decides as {}. SphNN decides the task as {} and denies ChatGPT\'s explanation after {} time of feedback.'.format(self.chatgpt_decision,
                                                                                                                                                        self.sphnn_decision, count)

                #self.add_to_log(answer, key=comment)
                self.log_json[count] = self.chatgpt_attempt
                with open(self.fw, 'w', encoding='utf8') as f:
                    json.dump(self.log_json, f)

                pprint(self.log_json)
                self.clear_log_json()
                break
            else:
                if self.sphnn_decision == self.chatgpt_decision:
                    self.feedback[count] = f""" You last answer is:"""+ self.chatgpt_attempt["chatgpt_raw_answer"]
                    self.feedback[count] += f"""\n your decision is correct!"""
                else:
                    self.feedback[count] = f""" You last answer is:""" + self.chatgpt_attempt["chatgpt_raw_answer"]
                    self.feedback[count] += f"""\n your decision is incorrect!"""

                if check_result == "HLU0":
                    self.feedback[count] = f""" You last answer is:""" + self.chatgpt_attempt["chatgpt_raw_answer"]
                    self.feedback[count] += f""" Explanation used partial information!"""
                elif check_result == "HLU1":
                    self.feedback[count] = f""" You last answer is:""" + self.chatgpt_attempt["chatgpt_raw_answer"]
                    self.feedback[count] += f""" Explanation used irrelevant information!"""
                elif check_result == "HLU2":
                    self.feedback[count] = f""" You last answer is:""" + self.chatgpt_attempt["chatgpt_raw_answer"]
                    self.feedback[count] += f""" Explanation contains reasoning flaws!"""
                elif check_result == "expl":
                    self.feedback[count] = f""" You last answer is:""" + self.chatgpt_attempt["chatgpt_raw_answer"]
                    self.feedback[count] += f""" Explanation is correct!"""

                self.chatgpt_attempt["sphnn_feedback"] = self.feedback[count].strip()
                self.log_json[count] = copy.deepcopy(self.chatgpt_attempt)
                self.chatgpt_last_explanation = copy.deepcopy(self.model_checker.TB_0)
                self.clear_chatgpt_attempt()

                prompt = self.prompt0 + self.feedback[count]
                count += 1
                self.log_json["sphnn_feedback_num"] = count
                time.sleep(TIME_BREAK)
                if NOT_IN_TEST:
                    answer = self.get_completion(prompt)
                else:
                    # answer = "cannot"
                    answer = "yes. (circle M0, inside, circle P), (circle P, inside, circle S), (circle S, overlaps, circle M0)"

    def make_ent_dic(self, edic):
        for key, value in edic.items():
            self.ndic[re.sub('\.n\.\d\d', '', value.upper())] = key

    def inital_prompt(self, task):
        task = [re.sub('\.n\.\d\d', '', s) for s in task]
        task = [make_sentence(s, lower = True) for s in task]
        self.prompt0 = f"""Your are a professional logician. \n
            Here is the instruction of your task: \n
                We represent '''all X are Y''' as circle X being inside circle Y, written as (circle X, inside, circle Y); \n
                             '''no X are Y''' as circle X disconnecting from circle Y, written as (circle X, outside, circle Y);\n
                             '''some X are Y''' as one of the three possible configurations:\n
                               (1) circle X is inside circle Y, written as (circle X, inside, circle Y); \n
                               (2) circle X partially overlaps with circle Y, written as (circle X, overlaps, circle Y); \n
                               (3) circle Y is inside circle X, written as (circle X, inside, circle Y);\n
                            '''some X are not Y''' as one of the three possible configurations:\n
                                (1) circle X disconnects from circle Y, written as (circle X, outside, circle Y); \n
                                (2) circle X properly contains circle Y, written as (circle X, inside, circle Y); \n
                                (3) circle X partially overlaps with circle Y, written as (circle X, overlaps, circle Y);\n
                Please note: if '''all X are Y''', then '''some X are Y''';\n
                Please note: Do not reply both '''cannot''' and '''yes''';\n
                Are the three statements '''{'. '.join(task)}''' compatible? Can you represent their relations using three circles as described above? \n
                If they cannot be represented by relations among three circles, reply '''cannot''', otherwise, reply '''yes'''; \n
                If you answer 'cannot', please explain what relations among the three circles are not possible and write each relation in the triple form, e.g, (circle X, inside, circle Y); \n
                If you answer 'yes', please explain the relations among the three circles in the list of triple forms, e.g., (circle X, inside, circle Y);\n
                Here is an example: your task is 'Are the three statements '''no M0 are S', 'all P are M0', 'some S are not P''' compatible? Can you represent their relations using three circles as described above?'; \n 
                                    your answer is: '''yes. (circle P, inside, circle M0), (circle M0, disconnect, circle S), (circle S, disconnect, circle P).'''\n
                You have enough time for this task. Please describe circles relations first. \n 
                Please make your answer as short as possible. \n 
                Please note: statements in most of the tasks are compatible. \n 
        """
        return self.prompt0

    def deduction_with_model_checker(self, data_dir="data/Syllogism/",
                                     fname= "Possible_Syllogism_3.txt", max_num = 2, M=2, randSym = False):
        '''
        sphNN helps ChatGPT to solve SAT problems.
        :param sphNN:
        :param ques:
        :return:
        '''
        d = ValidSyllogism(data_dir=data_dir, file_name = fname, use_random_symbol=randSym)
        ##done_lst = get_failed_cases(fdir="../data/hsphnn_runs/ChatGPT_SphNN/")

        for i in range(len(d.raw_data_list)):
            #if i >3: continue
            prompt = self.inital_prompt(d.raw_data_list[i])
            self.make_ent_dic(d.id2ent_dict_list[i])
            self.task = {
                "num":i,
                "data":d.data_list[i],
                "raw":d.raw_data_list[i]}
            self.log_json["task"] = self.task
            self.log_json["id"] = "log_data_{}.json".format(self.task["num"])
            self.fw = self.model_checker.save_dir + "log_data_{}.json".format(self.task["num"])
            #self.add_to_log(d.raw_data_list[i], key='the original task:')
            self.initialise_data(d.id2ent_dict_list[i])
            self.prompt_deduction_with_model_checker(prompt, max_num=max_num, Mepoch = M, randSym = randSym)


if __name__ == '__main__':
    with open('config/chatgpt_key.txt', 'r') as api_key:
        API_KEY = api_key.read().strip()
        with open('params.json', 'r') as ijh:
            params = json.load(ijh)
            for tblock in params["exp2"]["all_4o10F"]:
                data_dir = tblock["data_dir"]
                output_dir = tblock["output_dir"]
                Mepoch = tblock["M"]
                FileName = tblock["test_file"]
                vChatGPT = tblock["chatgpt"]
                randomSym = tblock["use_random"]
                result = tblock["result_file"]
                maxNum = tblock["max_num"]

                adualmind = GPT_SphNN_dualMind(API_KEY, vChatGPT, 'deductor')
                adualmind.model_checker.save_dir = output_dir
                adualmind.deduction_with_model_checker(data_dir = data_dir, fname = FileName,
                                                       max_num = maxNum, M = Mepoch, randSym = randomSym)

