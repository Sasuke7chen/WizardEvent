import os
import pdb
import random

import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import json
from transformers import AutoTokenizer
import string

REL_MAP = {'Precedence': 'after', 'Succession': 'before', 'Synchronous': 'simultaneously', 'Reason': 'cause', 'Result': 'effect', 'Condition': 'isCondition', 'Contrast': 'isContrast', 'Instantiation': 'isInstantiation', 'Restatement': 'isRestatement', 'Alternative': 'isAternative', 'Exception': 'isException'}
with open('/data1/tzw/data/explicit/clustering.json') as f: et_clusters = json.load(f)


def capitalize_sentence(sentence):
    # Capitalize the first letter
    capitalized = sentence.capitalize()
    
    # Check if the sentence ends with punctuation
    if not capitalized.endswith(tuple(string.punctuation)):
        # Add a period at the end
        capitalized += "."
    
    return capitalized
import re

def capitalize_sentences(input_string):
    # 使用句号、感叹号和问号来拆分句子
    sentences = re.split(r'(?<=[.!?])\s+', input_string)
     
    # 对每个句子应用capitalize()函数
    capitalized_sentences = [sentence.capitalize() for sentence in sentences]
    
    # 将句子重新组合成一个字符串
    capitalized_string = ' '.join(capitalized_sentences)
    if not capitalized_string.endswith(tuple(string.punctuation)):
        # Add a period at the end
        capitalized_string += "."
    return capitalized_string

def process_question_and_choices(question, choices):
    # Split the question into words
    words = question.split()

    # Extract the second and the last word (e0 and e1)
    if len(words) >= 2:
        e0 = words[1]
        e1 = words[-1][:-1]
    else:
        return "Invalid question format", []

    # Define the new question and choices templates based on the input choices
    if choices == ["Causes", "IsResult", "Vague"]:
        new_question = f"Which is the causal relationship between {e0} and {e1}?"
        new_choices = [f"{e0} causes {e1}.", f"{e0} is result of {e1}.", f"There is no obvious causal relationship between {e0} and {e1}."]
    elif choices == ["Before", "After", "Vague"]:
        new_question = f"Which is the temporal relationship between {e0} and {e1}?"
        new_choices = [f"{e0} is before {e1}.", f"{e0} is after {e1}.", f"There is no obvious temporal relationship between {e0} and {e1}."]
    elif choices == ["HasSubevent", "IsSubevent", "Vague"]:
        new_question = f"Which is the subordinate relationship between {e0} and {e1}?"
        new_choices = [f"{e1} is subevent of {e0}.", f"{e0} is subevent of {e1}.", f"There is no obvious subordinate relationship between {e0} and {e1}."]
    else:
        return "Invalid choices format", []

    return new_question, new_choices


def find_and_sample(A, B):
    # 替换A中所有子列表中的字符串中的空格为下划线
    A = [[s.replace(' ', '_') for s in sublist] for sublist in A]

    # 找到B中每个字符串在A的子列表中的索引
    indexes = set()
    for b in B:
        for i, sublist in enumerate(A):
            if b in sublist:
                indexes.add(i)
                break

    # 对于A中的每个子列表，如果其索引不在找到的索引列表中，则随机选择一个字符串
    C = []
    for i, sublist in enumerate(A):
        if i not in indexes:
            for j in sublist:
                C.append(j)
    C=random.sample(C, 20-len(B))
    # 将B中的内容随机插入到C中
    for b in B:
        insert_index = random.randint(0, len(C))
        C.insert(insert_index, b)

    return C

class IEREDataset(Dataset):
    def __init__(self,
            task_name,
            data_path,
            generative_format,
            use_cot=False,
            max_seq_length=512,
            k=None,
            i=None,
            shuffle=False,
            split=False,
            ):
        
        self.generative_format = generative_format
        self.i = i
        self.k = k

        if split:
            path = os.path.join(data_path, f'train_{i}.jsonl')
            # path = os.path.join(data_path, f'test{i}.jsonl')
        else:
            path = os.path.join(data_path, f'test.jsonl')
        path = path.replace('lmmc', 'mc')


        with open(path, 'r') as f:
            dataset = [json.loads(d) for d in f.readlines()]
            # dataset = [json.loads(d) for d in f.readlines()][:500]

        train_path = os.path.join(data_path, f'train.jsonl')
        with open(train_path, 'r') as f:
            self.trainset = [json.loads(d) for d in f.readlines()]
            # if shuffle:
            #     random.shuffle(self.trainset)

        self.task_name = task_name
        self.use_cot = use_cot

        self.formats = {
                'lmmc': 
                    {
                    'DTFit': self.lmmc_DTFit,
                    'HardSimExt': self.lmmc_HardSimExt,
                    },
                'mc': 
                    {
                    'DTFit': self.mc_DTFit,
                    'DTFit_json': self.mc_DTFit_json,
                    'HardSimExt': self.mc_HardSimExt,
                    'HardSimExt_json': self.mc_HardSimExt_json,
                    'SocialIQA': self.mc_SocialIQA,
                    'COPA': self.mc_COPA,
                    'STC': self.mc_STC,
                    'ECARE': self.mc_ECARE,
                    'ET': self.mc_ET,
                    'TIMETRAVEL': self.mc_TIMETRAVEL,
                    'MCTACO': self.mc_MCTACO,
                    'COPA': self.mc_COAP,
                    'SCITE': self.mc_SCITE,
                    'MCNC': self.mc_MCNC,
                    'MCNC_json': self.mc_MCNC_json,
                    'Claret': self.mc_semi,
                    'Finance': self.mc_semi,
                    'ESTER': self.mc_ESTER,
                    "explicit" : self.mc_explicit,
                    "explicit_schema" : self.mc_explicit_schema,
                    "explicit_instance" : self.mc_explicit_instance,
                    "explicit_instance_et" : self.mc_explicit_instance_et,
                    "explicit_rq3" : self.mc_explicit_rq3,
                    },
                're':
                    {
                    'ECARE': self.re_ECARE,
                    'MCTACO': self.re_MCTACO,
                    'SocialIQA': self.re_SocialIQA,
                    'ESTER': self.re_ESTER,
                    'MATRES':self.re_MATRES,
                    'ESL_D':self.re_ESL_D,
                    "explicit": self.re_explicit,
                    "explicit_schema": self.re_explicit_schema,
                    "explicit_instance" : self.re_explicit_instance,
                    "explicit_instance_et" : self.re_explicit_instance_et,
                    "explicit_rq3" : self.re_explicit_rq3,
                    },
                'nli':
                    {
                    'ECARE': self.nli_ECARE,
                    'TRACIE': self.nli_TRACIE,
                    'MCTACO': self.nli_MCTACO_v2,
                    'ALTLEX': self.nli_ALTLEX,
                    'TIMETRAVEL': self.nli_TIMETRAVEL,
                    'SocialIQA': self.nli_SocialIQA,
                    'ESTER': self.nli_ESTER,
                    },
                
                'gen': 
                    {
                    'ECARE': self.gen_ECARE,
                    'TellMeWhy': self.gen_TellMeWhy,
                    'WikiWhy': self.gen_WikiWhy,
                    'MCTACO': self.gen_MCTACO,
                    'SocialIQA': self.gen_SocialIQA,
                    'CQA': self.gen_CQA_v2,
                    'ESTER': self.gen_ESTER_v2,
                    'TIMETRAVEL': self.gen_TIMETRAVEL,
                    'SE2020T5_EQA': self.gen_SE2020T5_EQA,
                    'StoryGen': self.gen_StoryGen,
                    'MCNC': self.gen_MCNC,
                    'MCNC_json': self.gen_MCNC_json,
                    'HumanGeneral': self.gen_Human,
                    'HumanFinance': self.gen_Human,
                    'HumanMilitary': self.gen_Human,
                    },
                'eae':
                    {
                    'EAE': self.eae_EAE,
                    },

                }

        self.src_inputs_list = []
        self.labs = []
        self.src_list = []
        self.pos_list = []
        self.stm_list = []

        ind = 0
        for n, data in tqdm(enumerate(dataset)):

            data_generator = self.formats[generative_format][task_name](data)

            for src, stms, lab in data_generator:

                if src is None:
                    continue

                src_list = []
                stm_list = []
                for src_, stm_ in zip(src, stms):
                    if ind < 5: 
                        print('================src================')
                        print(src_)
                        print('================lab================')
                        print(lab)
                        print()

                    src_list.append(src_)
                    stm_list.append(stm_)

                self.src_list.append(src_list)
                self.stm_list.append(stm_list)
                self.labs.append(lab)
                ind += 1


    def build_demonstrations(self, doc_indices):

        seq = ''
        # seq = 'Examples:\n\n' # vicuna
        for ind in doc_indices:
            state = None
            ind_ = ind
            while state is None:
                state, _ = self.wrap_data(self.trainset[ind_], add_lab=True)
                ind_ += 1

            seq += state
            seq += '\n\n' 

        return seq


    def wrap_data(self, data, add_lab=False):
        if self.task_name == 'CQA':
            doc = data['doc']
            q = data['question']
            tgt = data['tgt']
            if tgt[-1] != '.':
                tgt += ' .'
            if not add_lab:
                src = f'Article: {doc}\nQuestion: {q}\nAnswer: ' 
            else:
                src = f'Article: {doc}\nQuestion: {q}\nAnswer: {tgt}' 

            return src, tgt


        elif self.task_name == 'SE2020T5_EQA':
            doc = data['doc']
            e0 = data['e0']
            e1 = data['e1']
            tgt = e0

            src = f"{doc} What prevented <e1> ? Answer: "

            if tgt[-1] != '.':
                tgt += ' .'

            if add_lab:
                src += f'{tgt}' 

            return src, tgt


        elif self.task_name == 'StoryGen':
            src = data['src']
            tgt = data['tgt']

            # unified prompt
            # src = f"Events:\n{src}\nQuestion:\nWhat is the next event?\nThe answer is"
            src = f"Context:\n{src}\nQuestion:\nWhat is the next event?\nAnswer:"

            # if src[-1] != '.':
            #     src += ' .'
            if tgt[-1] != '.':
                tgt += ' .'

            if add_lab:
                src += f'{tgt}' 

            return src, tgt

        elif self.task_name == 'TIMETRAVEL':
            if self.generative_format == 'gen':
                premise = data['premise']
                beginning = data['beginning']
                counterfactual = data['counterfactual']
                if counterfactual[-1] == '.':
                    counterfactual = counterfactual[:-1]

                original_ending = data['original_ending']
                tgt = data['lab']

                src = f'Context:\n{premise}\nOriginal beginning:\n{beginning}\nOriginal ending:\n{original_ending}\nQuestion:\nIf the original beginning is changed by "{counterfactual}", what will be a new ending ? Only output a new ending consisting of three sentences and don\'t output the changed beginning and other text.\nAnswer:'

                # ins = f'Question:\nWhat is the new story ending consisting of three sentences?\nAnswer: '
                # mid = f'If the story ending changes to the text delimited by the angel brackets <{counterfactual}>' 
                # src = f"""Context:\ngiven the story premise delimited by the triple quotes \"\"\"{premise}\"\"\"; the story beginning delimited by the triple backticks ```{beginning}```; the story ending delimited by the triple dashes ---{original_ending}---"""
                # src = '\n'.join([src, mid, ins])

                # src = f'{premise} {beginning} {original_ending}\nquestion:\nWhat will happen after if {counterfactual}?'

                # src = f'Answer the question within three sentences based on the context.\nContext:\n{premise} {beginning}\nQuestion:\nWhat will be the counterfact if "{counterfactual}"?\nAnswer: '

                if self.use_cot:
                    src += "Let's think step by step."
                else:
                    if add_lab:
                        src += tgt

                return src, tgt

            elif self.generative_format == 'mc':
                premise = data['premise']
                beginning = data['beginning']
                counterfactual = data['counterfactual']
                if counterfactual[-1] == '.':
                    counterfactual = counterfactual[:-1]

                original_ending = data['original_ending']
                tgt = data['lab'][0]
                if random.random() < 0.5:
                    choices = [tgt, original_ending]
                    lab = 0
                else:
                    choices = [original_ending, tgt]
                    lab = 1

                src = f'Context:\n{premise} {beginning}\nQuestion:\nWhat will happen if "{beginning}" changes to "{counterfactual}"?\nChoices:\nA. {choices[0]}\nB. {choices[1]}\nThe answer is'
                return src, lab

            elif self.generative_format == 'nli':
                premise = data['premise']
                beginning = data['beginning']
                counterfactual = data['counterfactual']
                if counterfactual[-1] == '.':
                    counterfactual = counterfactual[:-1]

                original_ending = data['original_ending']
                tgt = data['lab'][0]
                choices = [original_ending, tgt]
                if random.random() < 0.5:
                    tgt = choices[0]
                    lab = 'False'
                else:
                    tgt = choices[1]
                    lab = 'True'

                src = f'Context:\n{premise} {beginning}\nQuestion:\nIf "{beginning}" changes to "{counterfactual}", is it true that "{tgt}" would happen?\nChoices:\nTrue\nFalse\nThe answer is '
                return src, lab
        
        elif self.task_name == 'DTFit':
            lab2ind = {0: 'A', 1: 'B'}
            doc0 = data['doc0']
            doc1 = data['doc1']
            lab = lab2ind[data['lab']]

            docs = [doc0.lower(), doc1.lower()]

            # prompt = 'Select the more plausible event from the following two events:\nA: <e0> \nB: <e1> \nAnswer: '
            # prompt = 'Select a correct event from the following two events:\nA: <e0> \nB: <e1> \nAnswer: '
            # prompt = 'Question: Which event is correct ?\nA. <e0> \nB. <e1> \nAnswer: '
            prompt = 'Question:\nWhich event is more likely to happen ?\nChoices:\nA. <e0> \nB. <e1> \nAnswer:\n'

            src = prompt.replace('<e0>', doc0).replace('<e1>', doc1)

            if self.use_cot:
                src += "Let's think step by step."
            else:
                if add_lab:
                    src += lab

            return src, lab
        
        elif self.task_name == 'HardSimExt':
            lab2ind = {0: 'A', 1: 'B'}

            doc = data['doc']
            e0 = data['e0']
            e1 = data['e1']
            lab = lab2ind[data['lab']]

            prompt = f'Question:\nWhich event is semantically similar to "{doc}" ?\nChoices:\nA. <e0> .\nB. <e1> .\nAnswer:\n' 

            src = prompt.replace('<e0>', e0).replace('<e1>', e1) 

            if self.use_cot:
                src += "Let's think step by step."
            else:
                if add_lab:
                    src += lab

            return src, lab

        elif self.task_name in ['Claret', 'Finance']:
            if self.generative_format == 'mc':
                qmap = {'after': 'What happened after "<e0>"?\n', 'before': 'What happened before "<e0>"',
                        'cause': 'What is the cause of "<e0>"?\n', 'effect': 'What is the result of "<e0>"?\n',
                        'isCondition': 'What is the condition of "<e0>"?\n', 'isContrast': 'What is the contrast of "<e0>"?\n',
                        'isInstantiation': 'What is the instantiation of "<e0>"?\n', 'isRestatement': 'What is the restatement of "<e0>"?\n',
                        'isAlternative': 'What is the alternative of "<e0>"?\n', 'isException': 'What is the exception of "<e0>"?\n'}

                if 'ctx' in data:
                    context = data['ctx']
                else:
                    context = data['context']

                lab = data['lab']
                choices = [capitalize_sentence(c) for c in data['choices']]
                e0 = capitalize_sentence(data['e0'])
                e1 = choices[lab]

                random.shuffle(choices)
                lab = choices.index(e1)

                r = data['r']
                # r = REL_MAP[data['r']]
                if r not in qmap:
                    return None, None

                if context:
                    prompt = f'Context:\n{context}\n'
                else:
                    prompt = ''
                prompt += f'Question:\n{qmap[r].replace("<e0>", e0).strip()}\n'

                inds = [char for char in string.ascii_letters if char.isupper()]
                cs = ''.join([f'{inds[i]}.{c}\n' for i, c in enumerate(choices)])
                prompt += f'Choices:\n{cs}'
                prompt += f'The answer is'
                src = prompt
                # src = f"Answer the question by selecting {','.join([inds[i] for i in range(len(choices))])}\n" + prompt
                tgt = choices[lab]

                if self.use_cot:
                    src += "Let's think step by step."
                else:
                    if add_lab:
                        src += f' {inds[lab]}'

                return src, inds[lab]

        elif self.task_name == 'HumanGeneral':
            rel = data['r']
            if rel == 'cause':
                doc = data['doc']
                q = data['q']
                e0 = data['e0']
                e1 = data['e1']
                tgt = [e0, e1][data['lab']]
                src = f'Question:\nWhat is the {q} of the "{doc}"?\nAnswer:'
            elif rel == 'prediction':
                src = data['src']
                tgt = data['tgt']

                src = f"Context:\n{src}\nQuestion:\nWhat is the next event?\nAnswer:"

                if tgt[-1] != '.':
                    tgt += ' .'
            else:
                context = data['context']
                question = data['question']
                choices = data['choices']
                lab = data['lab']
                tgt = choices[lab]

                src = f'Context:\n{context}\nQuestion:\n{question}'

            return src, tgt

        elif self.task_name in ['HumanFinance', 'HumanMilitary']:
            rel = data['r']
            if rel == 'cause':
                ctx = data['ctx']
                e0 = data['e0']
                tgt = data['e1']
                src = f'Context:\n{ctx}\nQuestion:\nWhat is the cause of the "{e0}"?\nAnswer:'
            elif rel == 'prediction':
                ctx = data['ctx']
                e0 = data['e0']
                tgt = data['e1']
                src = f'Context:\n{ctx}\nQuestion:\nWhat will happen after "{e0}"?\nAnswer:'

            else:
                ctx = data['ctx']
                e0 = data['e0']
                tgt = data['e1']
                src = f'Context:\n{ctx}\nQuestion:\nWhat is the intention of subject of "{e0}"?\nAnswer:'

            return src, tgt


        elif self.task_name in ['ET']:
            if self.generative_format == 'mc':
                inds = [char for char in string.ascii_letters if char.isupper()]
                lab2ind = {i: char for i, char in enumerate(inds)}

                context = data['context']
                event = data['event']
                choices = data['choices']
                lab = lab2ind[data['lab']]

                # unified prompt
                # src = f'Context:\n{context}\nQuestion:\nWhat is the event type of "{event}" in the context?\n'
                # ct = f'Choices:n'
                # for i in range(len(choices)):
                #     ct += f'{lab2ind[i]}. {choices[i]}\n'
                # src = src + ct +'The answer is'

                # ICL for wizardLM .. 
                # src = f'Question: {context} What is the event type of "{event}"? '

                # ICL for vicuna .. 
                src = f'{context} What is the event type of "{event}"? '
                # ct = f'Choices:'
                ct = ''
                for i in range(len(choices)):
                    ct += f'{lab2ind[i]}. {choices[i]}. '
                src = src + ct + 'The answer is'

                if self.use_cot:
                    src += "Let's think step by step."
                else:
                    if add_lab:
                        src += f' {lab}'

                return src, lab


        elif self.task_name in ['EAE']:
            if self.generative_format == 'eae':
                inds = [char for char in string.ascii_letters if char.isupper()]
                lab2ind = {i: char for i, char in enumerate(inds)}

                context = data['context']
                event = data['event']
                role = data['role']
                lab = data['lab']

                # unified prompt
                src = f'Context:\n{context}\nQuestion:\nWhat is the {role} of the event "{event}" in the context?\nAnswer:'

                if self.use_cot:
                    src += "Let's think step by step."
                else:
                    if add_lab:
                        src += f' {lab}'

                return src, lab


        elif self.task_name in ['ECARE']:
            if self.generative_format == 'mc':
                lab2ind = {0: 'A', 1: 'B'}

                doc = data['doc']
                q = data['q']
                e0 = data['e0']
                e1 = data['e1']
                lab = lab2ind[data['lab']]

                # # ChatGPT
                # src = f'Event:\n{doc}\nQuestion:\nWhat is the {q} of the event?\nChoices:\nA. {e0} \nB. {e1}\nAnswer:\n'
                # # vicuna
                # src = f'Which one of "A. {e0}" and "B. {e1}" is the {q} of the "{doc}"? The {q} of the "{doc}" is '
                # # ESP
                # src = f'{doc} Question: What is the {q} of the event?\nChoices:\nA. {e0} \nB. {e1}\nThe answer is'

                # unified prompt
                src = f'Question:\nWhat is the {q} of "{doc}"?\nChoices:\nA. {e0}\nB. {e1}\nThe answer is'

                if self.use_cot:
                    src += "Let's think step by step."
                else:
                    if add_lab:
                        src += f' {lab}'

                return src, lab
                # lab2ind = {0: 'A', 1: 'B'}

                # # doc = data['doc']
                # doc = data['e0']
                # q = data['dim']
                # e0 = data['choices'][0]
                # e1 = data['choices'][1]
                # lab = 'A' if data['lab'] == e0 else 'B'

                # # # ChatGPT
                # # src = f'Event:\n{doc}\nQuestion:\nWhat is the {q} of the event?\nChoices:\nA. {e0} \nB. {e1}\nAnswer:\n'
                # # # vicuna
                # # src = f'Which one of "A. {e0}" and "B. {e1}" is the {q} of the "{doc}"? The {q} of the "{doc}" is '
                # # # ESP
                # # src = f'{doc} Question: What is the {q} of the event?\nChoices:\nA. {e0} \nB. {e1}\nThe answer is'

                # # unified prompt
                # src = f'Question:\nWhat is the {q} of "{doc}"?\nChoices:\nA. {e0}\nB. {e1}\nThe answer is'

                # if self.use_cot:
                #     src += "Let's think step by step."
                # else:
                #     if add_lab:
                #         src += f' {lab}'

                # return src, lab

            elif self.generative_format == 'nli':
                e0 = data['e0']
                e1 = data['e1']
                q = data['q']
                lab = data['lab']

                lab_map = {'cause': 'reason', 'effect': 'result'}

                # ChatGPT
                src = f'Question:\nIs "{e1}" the {lab_map[q]} of "{e0}" ?\nChoices:\nTrue\nFalse\nThe answer is'
                # # vicuna
                # src = f'Which one of "A. {e0}" and "B. {e1}" is the {q} of the "{doc}"? The {q} of the "{doc}" is '
                # # ESP
                # src = f'Event: {doc}\nQuestion: What is the {q} of the event ?'

                if self.use_cot:
                    src += "Let's think step by step."
                else:
                    if add_lab:
                        src += lab

                return src, lab

            elif self.generative_format == 're':
                e0 = data['e0']
                e1 = data['e1']

                lab_map = {'cause': 'A', 'effect': 'B'}
                lab = lab_map[data['lab']]

                # ChatGPT
                # src = f'Question:\nWhat is the relation between "{e0}" and "{e1}" ?\nChoices:\nA. Reason\nB. Result\nThe answer is '
                # # vicuna
                # src = f'Which one of "A. {e0}" and "B. {e1}" is the {q} of the "{doc}"? The {q} of the "{doc}" is '
                # # ESP
                # src = f'Event: {doc}\nQuestion: What is the {q} of the event ?'

                # unifeid prompt
                src = f'Question:\nIs "{e1}" the reason or result of "{e0}"?\nChoices:\nA. Reason\nB. Result\nThe answer is'


                if self.use_cot:
                    src += "Let's think step by step."
                else:
                    if add_lab:
                        src += lab

                return src, lab


            else:
                doc = data['doc']
                q = data['q']
                e0 = data['e0']
                e1 = data['e1']
                tgt = [e0, e1][data['lab']]

                # # ChatGPT
                # src = f'Event:\n{doc}\nQuestion:\nWhat is the {q} of the event?\nChoices:\nA. {e0} \nB. {e1}\nAnswer:\n'
                # # vicuna
                # src = f'Which one of "A. {e0}" and "B. {e1}" is the {q} of the "{doc}"? The {q} of the "{doc}" is '
                # # ESP
                # src = f'Event: {doc}\nQuestion: What is the {q} of the event ?'

                # unified prompt
                # src = f'Question:\nWhat is the {q} of the "{doc}"?\nThe answer is'
                src = f'Question:\nWhat is the {q} of the "{doc}"?\nAnswer:'

                if self.use_cot:
                    src += "Let's think step by step."
                else:
                    if add_lab:
                        src += f'\n{tgt}'

                return src, tgt

        elif self.task_name in ['COPA']:
            if self.generative_format == 'mc':
                lab2ind = {0: 'A', 1: 'B'}

                doc = data['doc']
                q = data['q']
                e0 = data['e0']
                e1 = data['e1']
                lab = lab2ind[data['lab']]

                # # ChatGPT
                # src = f'Event:\n{doc}\nQuestion:\nWhat is the {q} of the event?\nChoices:\nA. {e0} \nB. {e1}\nAnswer:\n'
                # # vicuna
                # src = f'Which one of "A. {e0}" and "B. {e1}" is the {q} of the "{doc}"? The {q} of the "{doc}" is '
                # # ESP
                # src = f'{doc} Question: What is the {q} of the event?\nChoices:\nA. {e0} \nB. {e1}\nThe answer is'

                # unified prompt
                src = f'Question:\nWhat is the {q} of "{doc}"?\nChoices:\nA. {e0}\nB. {e1}\nThe answer is'

                if self.use_cot:
                    src += "Let's think step by step."
                else:
                    if add_lab:
                        src += f' {lab}'

                return src, lab

            elif self.generative_format == 'nli':
                e0 = data['e0']
                e1 = data['e1']
                q = data['q']
                lab = data['lab']

                lab_map = {'cause': 'reason', 'effect': 'result'}

                # ChatGPT
                src = f'Question:\nIs "{e1}" the {lab_map[q]} of "{e0}" ?\nChoices:\nTrue\nFalse\nThe answer is'
                # # vicuna
                # src = f'Which one of "A. {e0}" and "B. {e1}" is the {q} of the "{doc}"? The {q} of the "{doc}" is '
                # # ESP
                # src = f'Event: {doc}\nQuestion: What is the {q} of the event ?'

                if self.use_cot:
                    src += "Let's think step by step."
                else:
                    if add_lab:
                        src += lab

                return src, lab

            elif self.generative_format == 're':
                e0 = data['e0']
                e1 = data['e1']

                lab_map = {'cause': 'A', 'effect': 'B'}
                lab = lab_map[data['lab']]

                # ChatGPT
                # src = f'Question:\nWhat is the relation between "{e0}" and "{e1}" ?\nChoices:\nA. Reason\nB. Result\nThe answer is '
                # # vicuna
                # src = f'Which one of "A. {e0}" and "B. {e1}" is the {q} of the "{doc}"? The {q} of the "{doc}" is '
                # # ESP
                # src = f'Event: {doc}\nQuestion: What is the {q} of the event ?'

                # unifeid prompt
                src = f'Question:\nIs "{e1}" the reason or result of "{e0}"?\nChoices:\nA. Reason\nB. Result\nThe answer is'


                if self.use_cot:
                    src += "Let's think step by step."
                else:
                    if add_lab:
                        src += lab

                return src, lab


            else:
                doc = data['doc']
                q = data['q']
                e0 = data['e0']
                e1 = data['e1']
                tgt = [e0, e1][data['lab']]

                # # ChatGPT
                # src = f'Event:\n{doc}\nQuestion:\nWhat is the {q} of the event?\nChoices:\nA. {e0} \nB. {e1}\nAnswer:\n'
                # # vicuna
                # src = f'Which one of "A. {e0}" and "B. {e1}" is the {q} of the "{doc}"? The {q} of the "{doc}" is '
                # # ESP
                # src = f'Event: {doc}\nQuestion: What is the {q} of the event ?'

                # unified prompt
                # src = f'Question:\nWhat is the {q} of the "{doc}"?\nThe answer is'
                src = f'Question:\nWhat is the {q} of the "{doc}"?\nAnswer:'

                if self.use_cot:
                    src += "Let's think step by step."
                else:
                    if add_lab:
                        src += f'\n{tgt}'

                return src, tgt
            
        elif self.task_name == 'ESTER':
            if self.generative_format == 'mc':
                lab2ind = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

                context = data['context']
                question = data['question']
                choices = data['choices']
                lab = lab2ind[data['lab']]

                inds = [char for char in string.ascii_letters if char.isupper()]
                ipt = ''.join([f'{inds[i]}.{c}\n' for i, c in enumerate(choices)])

                # unified prompt
                src = f'Context:\n{context}\nQuestion:\n{question}\nChoices:\n{ipt}\nThe answer is'

                if self.use_cot:
                    src += "Let's think step by step."
                else:
                    if add_lab:
                        src += f' {lab}'

                return src, lab

            elif self.generative_format == 're':
                lab2ind = {0: 'A', 1: 'B'}

                e0 = data['e0']
                e1 = data['e1']
                context = data['context']
                choices = data['choices']
                lab = lab2ind[data['lab']]
                # src = f'Context:\n{context}\nQuestion:\n{question}\nChoices:\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nAnswer:\n'

                # unified prompt
                src = f'Context: \n{context}\nQuestion:\nIs "{e1}" the condition of "{e0}"?\nChoices:\nA. no\nB. yes\nThe answer is'


                if self.use_cot:
                    src += "Let's think step by step."
                else:
                    if add_lab:
                        src += f' {lab}'

                return src, lab

            elif self.generative_format == 'nli':

                e0 = data['e0']
                e1 = data['e1']
                lab = data['lab']
                context = data['context']

                src = f'Context:\n{context}\nQuestion:\nIs "{e1}" the condition of "{e0}"?\nChoices:\nTrue\nFalse\nThe answer is '

                if self.use_cot:
                    src += "Let's think step by step."
                else:
                    if add_lab:
                        src += lab

                return src, lab
            elif self.generative_format == 'gen':

                context = data['context']
                question = data['question']
                choices = data['choices']
                lab = data['lab']

                tgt = choices[lab]

                # unified prompt
                src = f'Context:\n{context}\nQuestion:\n{question}\nAnswer:'

                if self.use_cot:
                    src += "Let's think step by step."
                else:
                    if add_lab:
                        src += f' {tgt}'

                return src, tgt


        elif self.task_name in ['TellMeWhy', 'WikiWhy']:
            doc = data['doc']
            q = data['q']
            lab = data['lab']

            # # ChatGPT
            # src = f'Event:\n{doc}\nQuestion:\nWhat is the {q} of the event?\nChoices:\nA. {e0} \nB. {e1}\nAnswer:\n'
            # # vicuna
            # src = f'Which one of "A. {e0}" and "B. {e1}" is the {q} of the "{doc}"? The {q} of the "{doc}" is '
            # ESP
            src = f'Event: {doc}\nQuestion: {q}'

            if self.use_cot:
                src += "Let's think step by step."
            else:
                if add_lab:
                    src += lab

            return src, lab


        elif self.task_name in ['MCTACO']:
            if self.generative_format == 'mc':
                lab2ind = {0: 'A', 1: 'B'}

                ctx = data['context']
                q = data['q']
                choices = data['choices']
                e0 = choices[0]
                e1 = choices[1]
                lab = lab2ind[data['lab']]

                # # vicuna
                # src = f'Which one of "A. {e0}" and "B. {e1}" is the {q} of the "{doc}"? The {q} of the "{doc}" is '
                # # ESP
                # src = f'Context:\n{ctx}\nQuestion:\n{q}\nChoices:\nA. {e0} \nB. {e1}\nThe answer is '
                # # ChatGPT
                # src = f'Context:\n{ctx}\nQuestion:\n{q}\nChoices:\nA. {e0} \nB. {e1}\n'

                # unified prompt
                src = f'Context:\n{ctx}\nQuestion:\n{q}\nChoices:\nA. {e0} \nB. {e1}\nThe answer is'


                if self.use_cot:
                    src += "Let's think step by step"
                else:
                    if add_lab:
                        src += f' {lab}'

                return src, lab
            elif self.generative_format == 're':
                context = data['context']
                e0 = data['e0']
                e1 = data['e1']
                choices = data['choices']
                lab = data['lab']
                rel = choices[lab]

                lab_map = {'before': 'A', 'after': 'B'}
                lab = lab_map[rel]

                # unifeid prompt
                src = f'Context:\n{context}\nQuestion:\nDoes "{e1}" happen before or after "{e0}"?\nChoices:\nA. Before\nB. After\nThe answer is'


                if self.use_cot:
                    src += "Let's think step by step."
                else:
                    if add_lab:
                        src += lab

                return src, lab
            elif self.generative_format == 'nli':
                e0 = data['e0']
                e1 = data['e1']
                ctx = data['context']
                r = data['r']
                lab = 'A' if data['lab'] == 'False' else 'B'

                # ChatGPT
                src = f'Context:\n{ctx}\nQuestion:\nIs it true that "{e1}" happen {r} "{e0}" ?\nChoices:\nA. False \nB. True \nThe answer is '
                # # vicuna
                # src = f'Which one of "A. {e0}" and "B. {e1}" is the {q} of the "{doc}"? The {q} of the "{doc}" is '
                # # ESP
                # src = f'Event: {doc}\nQuestion: What is the {q} of the event ?'

                if self.use_cot:
                    src += "Let's think step by step."
                else:
                    if add_lab:
                        src += lab

                return src, lab
            else:
                ctx = data['context']
                q = data['q']
                tgt = data['lab']

                # # vicuna
                # src = f'Which one of "A. {e0}" and "B. {e1}" is the {q} of the "{doc}"? The {q} of the "{doc}" is '
                # # ESP
                # src = f'Context:\n{ctx}\nQuestion:\n{q}\nChoices:\nA. {e0} \nB. {e1}\nThe answer is '
                # # ChatGPT
                # src = f'Context:\n{ctx}\nQuestion:\n{q}'

                # unified prompt
                src = f'Context:\n{ctx}\nQuestion:\n{q}\nAnswer:'
                # src = f'Context:\n{ctx}\nQuestion:\n{q}'

                if self.use_cot:
                    src += "Let's think step by step"
                else:
                    if add_lab:
                        src += f'\n{tgt}'

                return src, tgt


        elif self.task_name == 'TRACIE':
            lab2ind = {0: 'A', 1: 'B'}
            # lab2ind = {0: 'F', 1: 'T'}

            context = data['context']
            hyp = data['hyp'].lower()
            lab = lab2ind[data['lab']]

            # prompt = 'Context:\n<ctx>\nQuestion:\nIs it true that <hyp> ?\nChoices:\nFalse \nTrue \nAnswer:\n'
            prompt = 'Context: <ctx>\nQuestion: Is it true that <hyp> ?\nA. False \nB. True \nOnly output A or B.\nAnswer: '

            src = prompt.replace('<ctx>', context).replace('<hyp>', hyp)


            if self.use_cot:
                src += "Let's think step by step."
            else:
                if add_lab:
                    src += lab

            return src, lab

        # elif self.task_name == 'TRACIE':
        #     lab2ind = {0: 'Entailment', 1: 'Contradiction'}
        #
        #     context = data['context']
        #     hyp = data['hyp'].lower()
        #     lab = lab2ind[data['lab']]
        #
        #     prompt = '<ctx> <hyp>'
        #
        #     src = prompt.replace('<ctx>', context).replace('<hyp>', hyp)
        #
        #     if self.use_cot:
        #         src += "Let's think step by step."
        #     else:
        #         if add_lab:
        #             src += lab
        #
        #     return src, lab

        # elif self.task_name == 'ALTLEX':
        #     lab2ind = {'None': 'A', 'CAUSAL': 'B'}
        #
        #     doc = data['doc']
        #     e0 = data['e0'].strip()
        #     e1 = data['e1'].strip()
        #     lab = lab2ind[data['lab']]
        #
        #
        #     prompt = "Are \"<e0>\" and \"<e1>\" causal related ?\nChoices:\nA. False\nB. True\nAnswer:"
        #
        #     src = prompt.replace('<e0>', e0).replace('<e1>', e1)
        #
        #
        #     if self.use_cot:
        #         src += "Let's think step by step."
        #     else:
        #         if add_lab:
        #             src += lab
        #
        #     return src, lab


        elif self.task_name == 'ALTLEX':
            # lab2ind = {'None': 'A', 'CAUSAL': 'B'}
            lab2ind = {'None': 'False', 'CAUSAL': 'True'}

            doc = data['doc']
            e0 = data['e0'].strip()
            e1 = data['e1'].strip()
            lab = lab2ind[data['lab']]


            prompt = "Are \"<e0>\" and \"<e1>\" causal related ?\nChoices:\nFalse\nTrue\nAnswer:"

            src = prompt.replace('<e0>', e0).replace('<e1>', e1)


            if self.use_cot:
                src += "Let's think step by step."
            else:
                if add_lab:
                    src += lab

            return src, lab


        elif self.task_name == 'SocialIQA':
            if self.generative_format == 'mc':
                lab2ind = {0: 'A', 1: 'B', 2: 'C'}

                context = data['context']
                question = data['question']
                choices = data['choices']
                lab = lab2ind[data['lab']]
                # src = f'Context:\n{context}\nQuestion:\n{question}\nChoices:\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nAnswer:\n'

                # unified prompt
                src = f'Context:\n{context}\nQuestion:\n{question}\nChoices:\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nThe answer is'


                if self.use_cot:
                    src += "Let's think step by step."
                else:
                    if add_lab:
                        src += f' {lab}'

                return src, lab
            elif self.generative_format == 're':
                lab2ind = {0: 'A', 1: 'B'}

                e0 = data['e0']
                e1 = data['e1']
                choices = data['choices']
                lab = lab2ind[data['lab']]
                # src = f'Context:\n{context}\nQuestion:\n{question}\nChoices:\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nAnswer:\n'

                # unified prompt
                src = f'Question:\nIs "{e1}" the intention of "{e0}"?\nChoices:\nA. False\nB. True\nThe answer is'


                if self.use_cot:
                    src += "Let's think step by step."
                else:
                    if add_lab:
                        src += f' {lab}'

                return src, lab

            elif self.generative_format == 'nli':

                e0 = data['e0']
                e1 = data['e1']
                lab = data['lab']

                src = f'Question:\nIs "{e1}" the intention of "{e0}"?\nChoices:\nTrue\nFalse\nThe answer is '

                if self.use_cot:
                    src += "Let's think step by step."
                else:
                    if add_lab:
                        src += lab

                return src, lab
            else:
                context = data['context']
                question = data['question']
                choices = data['choices']
                lab = data['lab']
                tgt = choices[lab]

                # src = f'Context:\n{context}\nQuestion:\n{question}'

                # unified prompt
                # src = f'Context:\n{context}\nQuestion:\n{question}\nThe answer is'
                src = f'Context:\n{context}\nQuestion:\n{question}\nAnswer:'

                if self.use_cot:
                    src += "Let's think step by step."
                else:
                    if add_lab:
                        src += f'\n{tgt}'

                return src, tgt



        elif self.task_name == 'MCNC':
            if self.generative_format == 'mc':
                lab2ind = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

                stem = data['stem']
                doc = '. '.join(stem)
                if doc[-1] != '.':
                    doc += ' .'
                choices = data['choices']
                lab = lab2ind[data['lab']]

                # src = f'Events:\n{doc}\nQuestion:\nWhat is the next event?\nChoices:\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nE. {choices[4]}\nAnswer:\n'

                # unified prompt
                src = f'Context:\n{doc}\nQuestion:\nWhat is the next event?\nChoices:\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nE. {choices[4]}\nThe answer is'

                if self.use_cot:
                    src += "Let's think step by step."
                else:
                    if add_lab:
                        src += lab

                return src, lab
            else:
                stem = data['stem']
                doc = '. '.join(stem)
                if doc[-1] != '.':
                    doc += ' .'
                choices = data['choices']
                lab = data['lab']
                tgt = choices[lab]

                # unified prompt
                # src = f'Events:\n{doc}\nQuestion:\nWhat is the next event?\nThe answer is'
                src = f'Context:\n{doc}\nQuestion:\nWhat is the next event?\nAnswer:'

                if self.use_cot:
                    src += "Let's think step by step."
                else:
                    if add_lab:
                        src += f'\n{tgt}'

                return src, tgt


        elif self.task_name == 'MCNC_json':

            def wrap_event_json(event):
                if 'predicate' in event:
                    event['action'] = event.pop('predicate')
                event = {k: v for k, v in event.items() if v is not None}
                e_str = json.dumps(event)
                return e_str

            if self.generative_format == 'mc':
                lab2ind = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

                stem = data['stem']
                doc = '. '.join(stem)
                if doc[-1] != '.':
                    doc += ' .'
                choices = data['choices']
                lab = lab2ind[data['lab']]

                src = f'Events:\n{doc}\nQuestion:\nWhat is the next event?\nChoices:\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nE. {choices[4]}\nAnswer:\n'

                # # unified prompt
                # src = f'Events:\n{doc}\nQuestion:\nWhat is the next event?\nChoices:\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nE. {choices[4]}\nThe answer is'

                if self.use_cot:
                    src += "Let's think step by step."
                else:
                    if add_lab:
                        src += lab

                return src, lab
            else:
                events = [wrap_event_json(e) for e in data['events']]
                choices = [wrap_event_json(e) for e in data['choices']]
                lab = data['lab']
                tgt = choices[lab]

                doc = ',\n'.join(events)

                # unified prompt
                # src = f'Events:\n{doc}\nQuestion:\nWhat is the next event?\nThe answer is'
                src = f'Context:\n{doc}\nQuestion:\nWhat is the next event?\nAnswer:'

                if self.use_cot:
                    src += "Let's think step by step."
                else:
                    if add_lab:
                        src += f'\n{tgt}'

                return src, tgt



        elif self.task_name == 'STC':
            lab2ind = {0: 'A', 1: 'B'}

            story = data['story']
            doc = ' '.join(story)
            e0 = data['e0']
            e1 = data['e1']
            lab = lab2ind[data['lab']]

            # src = f'Event:\n{doc}\nQuestion:\nWhat is the next event?\nA. {e0} \nB. {e1}\nAnswer:\n'

            # unified prompt
            src = f'Context:\n{doc}\nQuestion:\nWhat is the next event?\nA. {e0} \nB. {e1}\nThe answer is'

            if self.use_cot:
                src += "Let's think step by step."
            else:
                if add_lab:
                    src += f' {lab}'

            return src, lab

        elif self.task_name == 'SCITE':
            lab_map = {'CAUSAL': 'B', 'None': 'A'}
            doc = data['doc']
            e0 = data['e0']
            e1 = data['e1']
            e0_pos = data['e0_pos']
            e1_pos = data['e1_pos']
            lab = lab_map[data['lab']]

            src = f'Context:\n{doc}\nQuestion:\nWhat is the relation between "{e0}" and "{e1}" ?\nOptions:\nA. no relation \nB. causal\nThe answer is'

            if self.use_cot:
                src += "Let's think step by step."
            else:
                if add_lab:
                    src += lab
            
            return src, lab
        else:
            raise Exception('Not defined task.')

    def mc_explicit_schema(self, data):

        prompt = 'Instructions:\nAnswer the question by selecting A, B, C, D.\n\n'
        # prompt = 'Answer the question using only the letter label of the option.\n\n'

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem
        lab2ind = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        choices = data['choices']
        context = data['context']
        question = data['question']
        lab=lab2ind[choices.index(data['e1'])]
        inds = [char for char in string.ascii_letters if char.isupper()]
        ipt = ''.join([f'{inds[i]}. {c}\n' for i, c in enumerate(choices)])

        # unified prompt
        if context:
            src = f'Context:\n{context}\nQuestion:\n{question}\nChoices:\n{ipt}\nThe answer is'
        else:
            src = f'Question:\n{question}\nChoices:\n{ipt}\nThe answer is'

        # src, lab = self.wrap_data(data)
        prompt += src

        srcs = [prompt]

        yield srcs, [['A', 'B', 'C', 'D']], lab
    
    def mc_explicit_instance(self, data):

        formatted_string = "\n".join([f"{key}:\n{value}" for key, value in data["instances"].items()])
        formatted_string="Instances:\n"+formatted_string+"\n"
        
        context = data['context']
        if context:
            prompt = 'Instructions:\nAnswer the question by selecting A, B, C, D. Note that all events appearing in \"Context\", \"Question\", and \"Choices\" refer to the specific events described in \"Instances\".\n\n'+formatted_string+"\n\n"
        else:
            prompt = 'Instructions:\nAnswer the question by selecting A, B, C, D. Note that all events appearing in \"Question\", and \"Choices\" refer to the specific events described in \"Instances\".\n\n'+formatted_string+"\n\n"
        # prompt = 'Answer the question using only the letter label of the option.\n\n'

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem
        lab2ind = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        choices = data['choices']
        
        question = data['question']
        lab=lab2ind[choices.index(data['e1'])]
        inds = [char for char in string.ascii_letters if char.isupper()]
        ipt = ''.join([f'{inds[i]}. {c}\n' for i, c in enumerate(choices)])

        # unified prompt
        if context:
            src = f'Context:\n{context}\nQuestion:\n{question}\nChoices:\n{ipt}\nThe answer is'
        else:
            src = f'Question:\n{question}\nChoices:\n{ipt}\nThe answer is'

        # src, lab = self.wrap_data(data)
        prompt += src

        srcs = [prompt]

        yield srcs, [['A', 'B', 'C', 'D']], lab
    
    def mc_explicit_rq3(self, data):

        formatted_string = "\n".join([f"{key}:\n{value}" for key, value in data["instances"].items()])
        formatted_string="Instances:\n"+formatted_string
        
        prompt = "Instructions:\nIn a scenario explained by the \"Context\", the \"Question\" ask about selecting one event that has a certain event relationship with a particular event, from all possible tail events provided by the \"Choices\", and the \"Instances\" explain all the events in detail with a few sentences. Event semantic knowledge refers to the abstract event types to which specific events belong, and the relationships between these abstract event types. Please output the event semantic knowledge used in solving the following problem. Note that all possible abstract event types in the \"Schema\", and the relationships between abstract events include HasSubevent, IsSubevent, Before, After, Causes, and IsResult. For the tuple [event0, relation, event1], HasSubevent indicates that event1 is a subevent of event0, IsSubevent indicates that event0 is a subevent of event1, Before indicates that event0 occurs before event1, After indicates that event0 occurs after event1, Causes indicates that event0 causes event1, and IsResult indicates that event0 is the result of event1. Output in JSON format, don't generate other sentences."+"\n\n"
        # prompt = 'Answer the question using only the letter label of the option.\n\n'
        prompt += "Requirements:\nAbstract event types can only be chosen from \"Schema\", and the relationships of abstract event types can only be selected from HasSubevent, IsSubevent, Before, After, Causes, and IsResult. Follow the format in examples, output in JSONL format. The key \"event type\" should correspond to a value that is a dictionary with events as keys and their abstracted categories as values. The key \"event relation\" should correspond to a value that is a list of tuples [event0, relation, event1]. The relationships between events include HasSubevent, IsSubevent, Before, After, Causes, and IsResult.\n\n"
        
        # A = [['obstruct justice'], ['make decision'], ['alert'], ['eat breakfast', 'eat breakfast'], ['enhancing technique'], ['celebrate'], ['travel'], ['preparation material'], ['study'], ['conflict'], ['lost job'], ['shape dough', 'knead dough'], ['experience delay'], ['planting flowers'], ['purchase ingredients'], ['submit paper'], ['take break'], ['publication'], ['fly kite'], ['abandon'], ['becoming tired', 'tired', 'feel tired'], ['communication'], ['start moving'], ['shower', 'have shower'], ['get money'], ['slumber', 'sleep'], ['seek help', 'seeking assistance', 'seek assistance'], ['learn', 'learn how'], ['feeling excitement', 'excited'], ['burying cat', 'bury cat'], ['inhaling'], ['examination', 'evaluate'], ['perform'], ['play music', 'hear music'], ['agreeing with'], ['continue dancing'], ['play sport', 'play sports'], ['recover illness'], ['selecting casket', 'choosing casket'], ['exercise', 'exercising'], ['healthcare policy'], ['employment'], ['landscape gardening', 'landscaping garden'], ['conference presentation'], ['tie shoelaces'], ['engage in conversation'], ['check weather'], ['write poem', 'write poetry'], ['pursue education', 'receive education'], ['experience contentment'], ['plan trip'], ['make bread'], ['misunderstanding', 'confusion'], ['repair kite'], ['purchase snacks'], ['analyzing data', 'analyze data'], ['bake cake'], ['hearing news'], ['dry'], ['award ceremony'], ['making friends'], ['eating', 'eat'], ['playing video games'], ['community engagement'], ['passing class'], ['drink more'], ['paint artwork'], ['getting contract'], ['attend class'], ['watch nature'], ['escape consequences'], ['planning strategy'], ['prepare documentation'], ['dream'], ['create'], ['record'], ['climb tree'], ['drink refreshment'], ['academic collaboration'], ['preheat oven'], ['shop for groceries'], ['building cathedral'], ['analyse', 'analysing'], ['plead guilty'], ['digging hole'], ['read book'], ['achievement', 'accomplishment'], ['take a walk'], ['improve skill', 'improve skills'], ['swim'], ['floorspace'], ['purchase supplies'], ['hear singing'], ['observe change'], ['evening concludes'], ['submitting exam'], ['family expansion'], ['financial concerns'], ['feeling lonely'], ['reflect', 'think'], ['organize garden party'], ['follow recipe', 'following recipe'], ['repair vehicle'], ['bricks'], ['complete homework'], ['eat ice cream'], ['gaining weight'], ['lose consciousness'], ['obtain kite', 'get kite'], ['gain knowledge', 'gaining knowledge'], ['distraction'], ['decompose'], ['reacting emotionally'], ['use yeast'], ['protest begins'], ['prepare food', 'prepare meal'], ['kick feet'], ['fall'], ['common interests'], ['visit market'], ['stretching'], ['express information'], ['leave reception'], ['legal action initiation'], ['teach baking classes'], ['go outside'], ['overwhelmed'], ['play continue'], ['commit crime'], ['choose recipe'], ['depart honeymoon'], ['traveling abroad'], ['breathe'], ['daydreaming'], ['rest prepare'], ['contract termination'], ['feeling motivated', 'feel motivated'], ['doing housework'], ['attending commencement'], ['recovering from injury'], ['feel pain'], ['attend afterparty'], ['kill'], ['cooking'], ['receive feedback'], ['hunt'], ['acquiring sponsorship'], ['medication'], ['extinguishing lights'], ['cultural appreciation'], ['choose clothes'], ['legal complication'], ['cleaning'], ['have party'], ['take stand'], ['recover quickly'], ['forming relationships'], ['experience emotion'], ['get weapon'], ['drink energy drink'], ['avoid distractions'], ['revise'], ['becoming hungry'], ['celebrating success', 'celebrate success'], ['receiving financial aid'], ['study more'], ['entertainment break'], ['heal'], ['finishing projects'], ['sweating'], ['life milestone'], ['feel cold'], ['have dinner'], ['teach', 'teaching'], ['formalizing partnership'], ['publish book'], ['overlook'], ['open space'], ['destroy evidenceescape captivity'], ['social interaction'], ['provide assistance'], ['write grant proposal'], ['enjoy breeze'], ['shop'], ['miss deadline'], ['plan future'], ['argue case'], ['legislation'], ['stress relief'], ['physical endurance'], ['memorizing'], ['signing documents'], ['crying'], ['maintenance disruption'], ['withdraw money'], ['graduating'], ['consider consequences'], ['volunteering service'], ['inauguration celebration'], ['achieving academic recognition'], ['doubt', 'ignorance'], ['play'], ['economic impact'], ['feeling energized'], ['external acknowledgement'], ['schedule change'], ['inaugurate ceremony'], ['show rehabilitation'], ['get injuried'], ['brain'], ['receive applause'], ['explore'], ['happy', 'feel happy'], ['watch clouds'], ['dancing'], ['respond to questions'], ['attend seminar'], ['participate event'], ['application process'], ['expansion'], ['teach others'], ['recognize individuals'], ['pay'], ['cool down'], ['acknowledge support'], ['resolve conflict'], ['break law'], ['live'], ['improve health'], ['seek therapy'], ['compose'], ['grieve'], ['career planning'], ['run'], ['being ablebodied'], ['labour'], ['have string'], ['launch investigation'], ['seek entertainment'], ['relaxation'], ['reconciliation'], ['intensifying music'], ['get flour'], ['apathy'], ['destruction'], ['lung'], ['getting job'], ['design recipe'], ['wander'], ['reviewing news'], ['achieving perfection'], ['study recipe'], ['seek approval'], ['negotiating'], ['supporting infrastructure'], ['discuss performance'], ['close attention'], ['interact with audience'], ['pack up kite'], ['submit evidence'], ['return home'], ['physical exertion'], ['engage attention'], ['study sessions'], ['escape captivity'], ['feel discomfort'], ['plan vacation'], ['planning', 'plan'], ['become engrossed'], ['fallAsleep'], ['continuing education'], ['eat cake'], ['introduction'], ['organize materials'], ['buy'], ['move feet'], ['planning confusion'], ['pass out'], ['measure ingredients'], ['initiate project'], ['cdeact'], ['increasing skill'], ['cause disruption'], ['discuss lessons'], ['disappointed with socializing'], ['meal'], ['receive communication'], ['cogitate'], ['loss'], ['emotional support'], ['common ground'], ['read'], ['delegating tasks'], ['overlooking data'], ['sing'], ['realization'], ['escalation of disputes'], ['submit assignment'], ['understanding'], ['die'], ['understanding better'], ['dance celebration'], ['switch roles'], ['being hired'], ['seek employment'], ['prepare breakfast'], ['evaluation knowledge'], ['healthcare demand'], ['personal satisfaction'], ['decorate cake'], ['execute action'], ['isolation'], ['abolish'], ['mix ingredients', 'prepare ingredients', 'get ingredients'], ['studying'], ['exhale'], ['watch kite'], ['tie shoes'], ['accept'], ['summarize'], ['being rejected'], ['anxiety'], ['sharing knowledge'], ['receive award'], ['change of interest'], ['pack supplies'], ['landscaping'], ['financial decision'], ['renew'], ['repare oven'], ['lose balance'], ['being nice'], ['plan travel'], ['distract'], ['improving technique', 'improving techniques'], ['incite conflict'], ['rest after activity'], ['refreshment'], ['talk', 'discuss'], ['eat snack'], ['sadness'], ['listen', 'remember'], ['exhaustion'], ['use breadmaker'], ['implement policy'], ['develop skills'], ['coordinate'], ['indifference'], ['sick'], ['invent'], ['apply'], ['discover'], ['resting', 'rest'], ['trust'], ['observation'], ['implement'], ['help'], ['ponder'], ['source ingredients'], ['leave event'], ['construct'], ['learning'], ['inquiry'], ['feeling invigorated'], ['adjust kite'], ['review'], ['investigate'], ['fall asleep'], ['disinterest'], ['discuss strategy'], ['physical exercise'], ['disappointment'], ['gather information'], ['prepare equipment'], ['cook'], ['disagreement'], ['take exam'], ['prepare presentation'], ['observe weather'], ['breathing'], ['misfortune']]
        A = et_clusters
        B = list(data["event_type"].values())
        combined = find_and_sample(A, B)
        schema = ", ".join(combined)
        prompt += "Schema:\n"+schema+"\n\n"
        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem
        prompt += "Here are some examples:\n\n"+"Instances:\nevent53:\nThis revelation ultimately prompted an individual in the courtroom audience to discretely exit the room and later that evening, the same individual, driven by fear of exposure, went on to commit a fatal assault against a witness who could connect him to the crime.\
\nevent67:\nWhile Mr. Smith was providing his account, he mentioned a key detail that was previously overlooked—a unique tattoo that he glimpsed on the perpetrator's arm.\
\nevent91:\nTwo weeks later, during the heated court proceedings at the downtown courthouse, the homeowner, Mr. Smith, was called to testify before the jury regarding the night of the incident.\
\nevent64:\nIn a quiet suburban neighborhood, a burglary occurred at the Smith residence, where an unknown assailant broke in and stole valuable heirlooms late at night.\
\nevent51:\nUpon hearing the new testimony about the tattoo, a juror who happened to have an interest in body art quietly made a mental note to research the design's origins, intrigued by its possible cultural significance.\
\nevent13:\nUpon hearing Mr. Smith's testimony, one juror with claustrophobia experienced a severe panic attack, which led to the court session being temporarily adjourned as the juror was rushed to the hospital for medical attention.\
\nevent90:\nIn a surprising turn of events during the tea break, the court stenographer, overwhelmed by guilt, approached the judge and admitted to tampering with court transcripts in a previous unrelated case, sparking an investigation into judicial misconduct.\
\n\nContext:\n\"event64\" is before \"event91\". \"event67\" is a subevent of \"event91\".\
\nQuestion:\nWhich is caused by \"event67\"?\
\nChoices:\nA. event13\
\nB. event90\
\nC. event53\
\nD. event51\
\n\nEvent type and event relation:\n{\"event_type\": {\"event67\": \"talk\", \"event53\": \"kill\", \"event64\": \"commit_crime\", \"event91\": \"take_stand\", \"event51\": \"hide_evidence\", \"event13\": \"escape\", \"event90\": \"confess\"}, \"event_relation\": [[\"commit_crime\", \"Before\", \"take_stand\"], [\"take_stand\", \"HasSubevent\", \"talk\"], [\"talk\", \"Causes\", \"kill\"]]}"+"\n\n"
        # prompt += "Examples:\n"+"""{'event_type': {'event67': 'talk', 'event53': 'kill', 'event64': 'commit_crime', 'event91': 'take_stand'}, 'event_relation': [['event64', 'Before', 'event91'], ['event91', 'HasSubevent', 'event67'], ['event67', 'Causes', 'event53']]}"""+"\n\n"
         
        prompt += "Now, based on the above, please output the event semantic knowledge used in solving the following problem.\n\n"                                                       
        prompt += formatted_string+'\n\n'
        
        choices = data['choices']
        context = data['context']
        question = data['question']
        lab = {}
        lab["event_type"]=data["event_type"]
        lab["event_relation"]=data["event_relation"]
        inds = [char for char in string.ascii_letters if char.isupper()]
        ipt = ''.join([f'{inds[i]}. {c}\n' for i, c in enumerate(choices)])

        # unified prompt
        src = f'Context:\n{context}\nQuestion:\n{question}\nChoices:\n{ipt}\nEvent type and event relation:'
        # src, lab = self.wrap_data(data)
        prompt += src

        srcs = [prompt]

        yield srcs, [['A', 'B', 'C', 'D']], json.dumps(lab)
    
    def re_explicit_schema(self, data):

        prompt = 'Instructions:\nAnswer the question by selecting A, B or C.\n\n'
        # vicuna
        # prompt = 'Determine the type of causal relationship between events by returning A, B or C.\n\n'

        # prompt = 'Answer the question using only the letter label of the option.\n\n'

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem

        # choices = data['choices']
        lab_map={"HasSubevent":"A", "IsSubevent":"B","Before":"A",  "After":"B","Causes":"A", "IsResult":"B","Vague" : "C"}
        choices = data['choices']
        context = data['context']
        question = data['question']
        # pdb.set_trace()
        
        new_question, new_choices=process_question_and_choices(question,choices)
        inds = [char for char in string.ascii_letters if char.isupper()]
        ipt = ''.join([f'{inds[i]}. {c}\n' for i, c in enumerate(new_choices)])
        
        lab=lab_map[data['re1']]
        # src, lab = self.wrap_data(data)
        if context:
            src=f'Context:\n{context}\nQuestion:\n{new_question}\nChoices:\n{ipt}\nThe answer is'
        else:
            src=f'Question:\n{new_question}\nChoices:\n{ipt}\nThe answer is'

        prompt += src

        srcs = [prompt]

        yield srcs, [new_choices], lab

    def re_explicit_instance(self, data):
        
        
        formatted_string = "\n".join([f"{key}:\n{value}" for key, value in data["instances"].items()])
        formatted_string="Instances:\n"+formatted_string+"\n"
        
        context = data['context']

        if context:
            prompt = 'Instructions:\nAnswer the question by selecting A, B or C. Note that all events appearing in \"Context\", \"Question\", and \"Choices\" refer to the specific events described in \"Instances\".\n\n'+formatted_string+"\n\n"
        else:
            prompt = 'Instructions:\nAnswer the question by selecting A, B or C. Note that all events appearing in \"Question\", and \"Choices\" refer to the specific events described in \"Instances\".\n\n'+formatted_string+"\n\n"

        # vicuna
        # prompt = 'Determine the type of causal relationship between events by returning A, B or C.\n\n'

        # prompt = 'Answer the question using only the letter label of the option.\n\n'

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem

        # choices = data['choices']
        lab_map={"HasSubevent":"A", "IsSubevent":"B","Before":"A",  "After":"B","Causes":"A", "IsResult":"B","Vague" : "C"}
        choices = data['choices']
        
        question = data['question']
        new_question, new_choices=process_question_and_choices(question,choices)
        inds = [char for char in string.ascii_letters if char.isupper()]
        ipt = ''.join([f'{inds[i]}. {c}\n' for i, c in enumerate(new_choices)])
        
        lab=lab_map[data['re1']]
        # src, lab = self.wrap_data(data)
        if context:
            src=f'Context:\n{context}\nQuestion:\n{new_question}\nChoices:\n{ipt}\nThe answer is'
        else:
            src=f'Question:\n{new_question}\nChoices:\n{ipt}\nThe answer is'


        prompt += src

        srcs = [prompt]

        yield srcs, [new_choices], lab
    
    def mc_explicit_instance_et(self, data):

        ets = data['event_type']

        formatted_string = "\n".join([f"{key}:\n{value}" for key, value in data["instances"].items()])
        formatted_string="Instances:\n"+formatted_string+"\n"
        
        context = data['context']
        if context:
            prompt = 'Instructions:\nAnswer the question by selecting A, B, C, D. Note that all events appearing in \"Context\", \"Question\", and \"Choices\" refer to the specific events described in \"Instances\".\n\n'+formatted_string+"\n\n"
        else:
            prompt = 'Instructions:\nAnswer the question by selecting A, B, C, D. Note that all events appearing in \"Question\", and \"Choices\" refer to the specific events described in \"Instances\".\n\n'+formatted_string+"\n\n"
        # prompt = 'Answer the question using only the letter label of the option.\n\n'

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem
        lab2ind = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        choices = data['choices']
        
        question = data['question']
        lab=lab2ind[choices.index(data['e1'])]
        inds = [char for char in string.ascii_letters if char.isupper()]
        ipt = ''.join([f'{inds[i]}. {c}\n' for i, c in enumerate(choices)])

        # unified prompt
        if context:
            src = f'Context:\n{context}\nQuestion:\n{question}\nChoices:\n{ipt}\nThe answer is'
        else:
            src = f'Question:\n{question}\nChoices:\n{ipt}\nThe answer is'

        # src, lab = self.wrap_data(data)
        prompt += src

        for eid, et in ets.items():
            prompt = prompt.replace(eid, et)

        srcs = [prompt]

        yield srcs, [['A', 'B', 'C', 'D']], lab
    

    def re_explicit_instance_et(self, data):
        
        ets = data['event_type']
        
        formatted_string = "\n".join([f"{key}:\n{value}" for key, value in data["instances"].items()])
        formatted_string="Instances:\n"+formatted_string+"\n"
        
        context = data['context']

        if context:
            prompt = 'Instructions:\nAnswer the question by selecting A, B or C. Note that all events appearing in \"Context\", \"Question\", and \"Choices\" refer to the specific events described in \"Instances\".\n\n'+formatted_string+"\n\n"
        else:
            prompt = 'Instructions:\nAnswer the question by selecting A, B or C. Note that all events appearing in \"Question\", and \"Choices\" refer to the specific events described in \"Instances\".\n\n'+formatted_string+"\n\n"

        # vicuna
        # prompt = 'Determine the type of causal relationship between events by returning A, B or C.\n\n'

        # prompt = 'Answer the question using only the letter label of the option.\n\n'

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem

        # choices = data['choices']
        lab_map={"HasSubevent":"A", "IsSubevent":"B","Before":"A",  "After":"B","Causes":"A", "IsResult":"B","Vague" : "C"}
        choices = data['choices']
        
        question = data['question']
        new_question, new_choices=process_question_and_choices(question,choices)
        inds = [char for char in string.ascii_letters if char.isupper()]
        ipt = ''.join([f'{inds[i]}. {c}\n' for i, c in enumerate(new_choices)])
        
        lab=lab_map[data['re1']]
        # src, lab = self.wrap_data(data)
        if context:
            src=f'Context:\n{context}\nQuestion:\n{new_question}\nChoices:\n{ipt}\nThe answer is'
        else:
            src=f'Question:\n{new_question}\nChoices:\n{ipt}\nThe answer is'


        prompt += src

        for eid, et in ets.items():
            prompt = prompt.replace(eid, et)

        srcs = [prompt]

        yield srcs, [new_choices], lab
    
    def re_explicit_rq3(self, data):
        formatted_string = "\n".join([f"{key}:\n{value}" for key, value in data["instances"].items()])
        formatted_string="Instances:\n"+formatted_string
        
        prompt = "Instructions:\nIn a scenario explained by the \"Context\", the \"Question\" ask about selecting one relationship between two events, from all possible relationships provided by the \"Choices\", and the \"Instances\" explain all the events in detail with a few sentences. Event semantic knowledge refers to the abstract event types to which specific events belong, and the relationships between these abstract event types. Please output the event semantic knowledge used in solving the following problem. Note that all possible abstract event categories in the \"Schema\", and the relationships between abstract events include HasSubevent, IsSubevent, Before, After, Causes, and IsResult. For the tuple [event0, relation, event1], HasSubevent indicates that event1 is a subevent of event0, IsSubevent indicates that event0 is a subevent of event1, Before indicates that event0 occurs before event1, After indicates that event0 occurs after event1, Causes indicates that event0 causes event1, and IsResult indicates that event0 is the result of event1."+"\n\n"
        # prompt = 'Answer the question using only the letter label of the option.\n\n'
        prompt += "Requirements:\nAbstract event types can only be chosen from \"Schema\", and the relationships of abstract event types can only be selected from HasSubevent, IsSubevent, Before, After, Causes, and IsResult. Follow the format in examples, output in JSONL format. The key \"event type\" should correspond to a value that is a dictionary with events as keys and their abstracted categories as values. The key \"event relation\" should correspond to a value that is a list of tuples [event0, relation, event1]. The relationships between events include HasSubevent, IsSubevent, Before, After, Causes, and IsResult.\n\n"
        
        # A = [['obstruct justice'], ['make decision'], ['alert'], ['eat breakfast', 'eat breakfast'], ['enhancing technique'], ['celebrate'], ['travel'], ['preparation material'], ['study'], ['conflict'], ['lost job'], ['shape dough', 'knead dough'], ['experience delay'], ['planting flowers'], ['purchase ingredients'], ['submit paper'], ['take break'], ['publication'], ['fly kite'], ['abandon'], ['becoming tired', 'tired', 'feel tired'], ['communication'], ['start moving'], ['shower', 'have shower'], ['get money'], ['slumber', 'sleep'], ['seek help', 'seeking assistance', 'seek assistance'], ['learn', 'learn how'], ['feeling excitement', 'excited'], ['burying cat', 'bury cat'], ['inhaling'], ['examination', 'evaluate'], ['perform'], ['play music', 'hear music'], ['agreeing with'], ['continue dancing'], ['play sport', 'play sports'], ['recover illness'], ['selecting casket', 'choosing casket'], ['exercise', 'exercising'], ['healthcare policy'], ['employment'], ['landscape gardening', 'landscaping garden'], ['conference presentation'], ['tie shoelaces'], ['engage in conversation'], ['check weather'], ['write poem', 'write poetry'], ['pursue education', 'receive education'], ['experience contentment'], ['plan trip'], ['make bread'], ['misunderstanding', 'confusion'], ['repair kite'], ['purchase snacks'], ['analyzing data', 'analyze data'], ['bake cake'], ['hearing news'], ['dry'], ['award ceremony'], ['making friends'], ['eating', 'eat'], ['playing video games'], ['community engagement'], ['passing class'], ['drink more'], ['paint artwork'], ['getting contract'], ['attend class'], ['watch nature'], ['escape consequences'], ['planning strategy'], ['prepare documentation'], ['dream'], ['create'], ['record'], ['climb tree'], ['drink refreshment'], ['academic collaboration'], ['preheat oven'], ['shop for groceries'], ['building cathedral'], ['analyse', 'analysing'], ['plead guilty'], ['digging hole'], ['read book'], ['achievement', 'accomplishment'], ['take a walk'], ['improve skill', 'improve skills'], ['swim'], ['floorspace'], ['purchase supplies'], ['hear singing'], ['observe change'], ['evening concludes'], ['submitting exam'], ['family expansion'], ['financial concerns'], ['feeling lonely'], ['reflect', 'think'], ['organize garden party'], ['follow recipe', 'following recipe'], ['repair vehicle'], ['bricks'], ['complete homework'], ['eat ice cream'], ['gaining weight'], ['lose consciousness'], ['obtain kite', 'get kite'], ['gain knowledge', 'gaining knowledge'], ['distraction'], ['decompose'], ['reacting emotionally'], ['use yeast'], ['protest begins'], ['prepare food', 'prepare meal'], ['kick feet'], ['fall'], ['common interests'], ['visit market'], ['stretching'], ['express information'], ['leave reception'], ['legal action initiation'], ['teach baking classes'], ['go outside'], ['overwhelmed'], ['play continue'], ['commit crime'], ['choose recipe'], ['depart honeymoon'], ['traveling abroad'], ['breathe'], ['daydreaming'], ['rest prepare'], ['contract termination'], ['feeling motivated', 'feel motivated'], ['doing housework'], ['attending commencement'], ['recovering from injury'], ['feel pain'], ['attend afterparty'], ['kill'], ['cooking'], ['receive feedback'], ['hunt'], ['acquiring sponsorship'], ['medication'], ['extinguishing lights'], ['cultural appreciation'], ['choose clothes'], ['legal complication'], ['cleaning'], ['have party'], ['take stand'], ['recover quickly'], ['forming relationships'], ['experience emotion'], ['get weapon'], ['drink energy drink'], ['avoid distractions'], ['revise'], ['becoming hungry'], ['celebrating success', 'celebrate success'], ['receiving financial aid'], ['study more'], ['entertainment break'], ['heal'], ['finishing projects'], ['sweating'], ['life milestone'], ['feel cold'], ['have dinner'], ['teach', 'teaching'], ['formalizing partnership'], ['publish book'], ['overlook'], ['open space'], ['destroy evidenceescape captivity'], ['social interaction'], ['provide assistance'], ['write grant proposal'], ['enjoy breeze'], ['shop'], ['miss deadline'], ['plan future'], ['argue case'], ['legislation'], ['stress relief'], ['physical endurance'], ['memorizing'], ['signing documents'], ['crying'], ['maintenance disruption'], ['withdraw money'], ['graduating'], ['consider consequences'], ['volunteering service'], ['inauguration celebration'], ['achieving academic recognition'], ['doubt', 'ignorance'], ['play'], ['economic impact'], ['feeling energized'], ['external acknowledgement'], ['schedule change'], ['inaugurate ceremony'], ['show rehabilitation'], ['get injuried'], ['brain'], ['receive applause'], ['explore'], ['happy', 'feel happy'], ['watch clouds'], ['dancing'], ['respond to questions'], ['attend seminar'], ['participate event'], ['application process'], ['expansion'], ['teach others'], ['recognize individuals'], ['pay'], ['cool down'], ['acknowledge support'], ['resolve conflict'], ['break law'], ['live'], ['improve health'], ['seek therapy'], ['compose'], ['grieve'], ['career planning'], ['run'], ['being ablebodied'], ['labour'], ['have string'], ['launch investigation'], ['seek entertainment'], ['relaxation'], ['reconciliation'], ['intensifying music'], ['get flour'], ['apathy'], ['destruction'], ['lung'], ['getting job'], ['design recipe'], ['wander'], ['reviewing news'], ['achieving perfection'], ['study recipe'], ['seek approval'], ['negotiating'], ['supporting infrastructure'], ['discuss performance'], ['close attention'], ['interact with audience'], ['pack up kite'], ['submit evidence'], ['return home'], ['physical exertion'], ['engage attention'], ['study sessions'], ['escape captivity'], ['feel discomfort'], ['plan vacation'], ['planning', 'plan'], ['become engrossed'], ['fallAsleep'], ['continuing education'], ['eat cake'], ['introduction'], ['organize materials'], ['buy'], ['move feet'], ['planning confusion'], ['pass out'], ['measure ingredients'], ['initiate project'], ['cdeact'], ['increasing skill'], ['cause disruption'], ['discuss lessons'], ['disappointed with socializing'], ['meal'], ['receive communication'], ['cogitate'], ['loss'], ['emotional support'], ['common ground'], ['read'], ['delegating tasks'], ['overlooking data'], ['sing'], ['realization'], ['escalation of disputes'], ['submit assignment'], ['understanding'], ['die'], ['understanding better'], ['dance celebration'], ['switch roles'], ['being hired'], ['seek employment'], ['prepare breakfast'], ['evaluation knowledge'], ['healthcare demand'], ['personal satisfaction'], ['decorate cake'], ['execute action'], ['isolation'], ['abolish'], ['mix ingredients', 'prepare ingredients', 'get ingredients'], ['studying'], ['exhale'], ['watch kite'], ['tie shoes'], ['accept'], ['summarize'], ['being rejected'], ['anxiety'], ['sharing knowledge'], ['receive award'], ['change of interest'], ['pack supplies'], ['landscaping'], ['financial decision'], ['renew'], ['repare oven'], ['lose balance'], ['being nice'], ['plan travel'], ['distract'], ['improving technique', 'improving techniques'], ['incite conflict'], ['rest after activity'], ['refreshment'], ['talk', 'discuss'], ['eat snack'], ['sadness'], ['listen', 'remember'], ['exhaustion'], ['use breadmaker'], ['implement policy'], ['develop skills'], ['coordinate'], ['indifference'], ['sick'], ['invent'], ['apply'], ['discover'], ['resting', 'rest'], ['trust'], ['observation'], ['implement'], ['help'], ['ponder'], ['source ingredients'], ['leave event'], ['construct'], ['learning'], ['inquiry'], ['feeling invigorated'], ['adjust kite'], ['review'], ['investigate'], ['fall asleep'], ['disinterest'], ['discuss strategy'], ['physical exercise'], ['disappointment'], ['gather information'], ['prepare equipment'], ['cook'], ['disagreement'], ['take exam'], ['prepare presentation'], ['observe weather'], ['breathing'], ['misfortune']]
        A = et_clusters
        B = list(data["event_type"].values())
        combined = find_and_sample(A, B)
        schema = ", ".join(combined)
        prompt += "Schema:\n"+schema+"\n\n"
        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem
        
        prompt += "Here are some examples:\n\n"+"Instances:\nevent53:\nThis revelation ultimately prompted an individual in the courtroom audience to discretely exit the room and later that evening, the same individual, driven by fear of exposure, went on to commit a fatal assault against a witness who could connect him to the crime.\
\nevent67:\nWhile Mr. Smith was providing his account, he mentioned a key detail that was previously overlooked—a unique tattoo that he glimpsed on the perpetrator's arm.\
\nevent91:\nTwo weeks later, during the heated court proceedings at the downtown courthouse, the homeowner, Mr. Smith, was called to testify before the jury regarding the night of the incident.\
\nevent64:\nIn a quiet suburban neighborhood, a burglary occurred at the Smith residence, where an unknown assailant broke in and stole valuable heirlooms late at night.\
\n\nContext:\n\"event64\" is before \"event91\". \"event67\" is a subevent of \"event91\".\
\nQuestion:\nWhich is the causal relationship between \"event67\" and \"event53\"?\
\nChoices:\nA. \"event67\" causes \"event53\".\
\nB. \"event67\" is result of \"event53\".\
\nC. There is no obvious causal relationship between \"event67\" and \"event53\".\
\n\nEvent type and event relation:\n{\"event_type\": {\"event67\": \"talk\", \"event53\": \"kill\", \"event64\": \"commit_crime\", \"event91\": \"take_stand\"}, \"event_relation\": [[\"commit_crime\", \"Before\", \"take_stand\"], [\"take_stand\", \"HasSubevent\", \"talk\"], [\"talk\", \"Causes\", \"kill\"]]}"+"\n\n"

        # prompt += "Examples:\n"+"""{'event_type': {'event67': 'talk', 'event53': 'kill', 'event64': 'commit_crime', 'event91': 'take_stand'}, 'event_relation': [['event64', 'Before', 'event91'], ['event91', 'HasSubevent', 'event67'], ['event67', 'Causes', 'event53']]}"""+"\n\n"
         
        prompt += "Now, based on the above, please output the event semantic knowledge used in solving the following problem.\n\n"
        prompt += formatted_string+'\n\n'
        
        choices = data['choices']
        context = data['context']
        question = data['question']
        new_question, new_choices=process_question_and_choices(question,choices)
        inds = [char for char in string.ascii_letters if char.isupper()]
        ipt = ''.join([f'{inds[i]}. {c}\n' for i, c in enumerate(new_choices)])
        lab = {}
        
        
        lab["event_type"]=data["event_type"]
        lab["event_relation"]=data["event_relation"]
        
        src=f'Context:\n{context}\nQuestion:\n{new_question}\nChoices:\n{ipt}\nEvent type and event relation:'
        prompt += src

        
        srcs = [prompt]

        yield srcs, [new_choices], json.dumps(lab)


    def mc_DTFit(self, data):

        # prompt = 'Answer the question by selecting A, B\n\n'

        # vicuna
        prompt = 'Answer the question by returning A or B.\n\n'

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem

        doc0 = data['doc0']
        doc1 = data['doc1']
        choices = [doc0.lower(), doc1.lower()]

        src, lab = self.wrap_data(data)
        prompt += src

        srcs = [prompt]

        yield srcs, [choices], lab

    def mc_DTFit_json(self, data):

        lab2ind = {0: 'A', 1: 'B'}

        d0 = data['d0']
        d1 = data['d1']
        lab = data['lab']

        def wrap_event_json(event):
            return str(event).lower()

        e0 = wrap_event_json(d0)
        e1 = wrap_event_json(d1)

        es = [e0, e1]

        # prompt = 'Select the more plausible event from the following two events:\nA: <e0> \nB: <e1> \nAnswer: '
        # prompt = 'Select a correct event from the following two events:\nA: <e0> \nB: <e1> \nAnswer: '
        # prompt = 'Question: Which event is correct ?\nA. <e0> \nB. <e1> \nAnswer: '
        prompt = 'Answer the question by selecting A, B\nQeustion:\nWhich event is more likely to happen ?\nChoices:\nA. <e0> \nB. <e1> \nAnswer:\n'

        src = prompt.replace('<e0>', e0).replace('<e1>', e1)

        srcs = [src]

        yield srcs, [es], lab2ind[lab]


    def mc_COPA(self, data):

        lab2ind = {0: 'A', 1: 'B'}

        doc = data['doc']
        q = data['q']
        e0 = data['e0']
        e1 = data['e1']
        lab = data['lab']

        src = f'Answer the question by selecting A, B.\nEvent:\n{doc}\nQuestion:\nWhat is the {q} of the event?\nA. {e0} \nB. {e1}\nAnswer:\n'
        es = [e0, e1]

        srcs = [src]

        yield srcs, [es], lab2ind[lab]

    def mc_semi(self, data):

        prompt = 'Answer the question by selecting A,B,C\n\n'

        dem = self.build_demonstrations(random.sample(list(range(50)), self.k))
        prompt += dem

        # e0 = data['e0']
        # e1 = data['e1']
        # choices = [e0, e1]
        choices = [capitalize_sentence(c) for c in data['choices']]

        src, lab = self.wrap_data(data)
        if src is None:
            return None, None, None

        prompt += src


        srcs = [prompt]

        yield srcs, [choices], lab

    def mc_ET(self, data):

        prompt = 'Answer the question by selecting from the choices.\n'

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem

        choices = data['choices']

        src, lab = self.wrap_data(data)
        prompt += src

        srcs = [prompt]

        yield srcs, choices, lab

    def eae_EAE(self, data):

        prompt = ''

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem

        src, lab = self.wrap_data(data)
        prompt += src

        srcs = [prompt]

        yield srcs, [[]], lab


    def mc_ECARE(self, data):
         # vicuna
        # prompt = 'Answer the question by returning A or B.\n\n'
        # prompt = ''

        prompt = 'Answer the question by selecting A or B.\n\n'
        # prompt = 'Answer the question using only the letter label of the option.\n\n'

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem

        e0 = data['choices'][0]
        e1 = data['choices'][1]
        choices = [e0, e1]

        doc = data['e0']
        q = data['dim']
        if data['lab'] == e0:
            lab = 'A'
        else:
            lab = 'B'
        src = f'Question:\nWhat is the {q} of "{doc}"?\nChoices:\nA. {e0}\nB. {e1}\nThe answer is:\n'
        prompt += src

        srcs = [prompt]

        yield srcs, [choices], lab

        # # vicuna
        # # prompt = 'Answer the question by returning A or B.\n\n'
        # # prompt = ''

        # prompt = 'Answer the question by selecting A, B\n\n'
        # # prompt = 'Answer the question using only the letter label of the option.\n\n'
        # # doc, e0, e1, q, lab(0, 1)
        # # e0, choices, dim, lab
        # dem = self.build_demonstrations(list(range(self.k)))
        # prompt += dem

        # # e0 = data['e0']
        # # e1 = data['e1']
        # # choices = [e0, e1]
        # choices = data['choices']

        # src, lab = self.wrap_data(data)
        # prompt += src

        # srcs = [prompt]

        # yield srcs, [choices], lab

    def gen_ECARE(self, data):

        # # vicuna
        # prompt = 'Answer the question by returning A or B.\n\n'

        prompt = ''

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem

        e0 = data['e0']
        e1 = data['e1']
        choices = [e0, e1]

        src, lab = self.wrap_data(data)
        prompt += src

        srcs = [prompt]

        yield srcs, [choices], lab

    

    def gen_Human(self, data):

        # vicuna
        # prompt = 'Answer the question by returning A or B.\n\n'

        prompt = ''

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem

        src, lab = self.wrap_data(data)
        prompt += src

        srcs = [prompt]

        yield srcs, [[]], lab

    def gen_TellMeWhy(self, data):

        # vicuna
        # prompt = 'Answer the question by returning A or B.\n\n'

        prompt = ''

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem

        src, lab = self.wrap_data(data)
        prompt += src

        srcs = [prompt]
        choices = [lab]

        yield srcs, [choices], lab

    def gen_WikiWhy(self, data):

        # vicuna
        # prompt = 'Answer the question by returning A or B.\n\n'

        prompt = ''

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem

        src, lab = self.wrap_data(data)
        prompt += src

        srcs = [prompt]
        choices = [lab]

        yield srcs, [choices], lab

    def nli_ECARE(self, data):

        # prompt = 'Answer the question by returning A or B.\n\n'

        prompt = 'Answer the question by returing True or False.\n\n'

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem

        e0 = data['e0']
        e1 = data['e1']
        choices = [e0, e1]

        src, lab = self.wrap_data(data)
        prompt += src

        srcs = [prompt]

        yield srcs, [choices], lab

    def re_ECARE(self, data):

        prompt = 'Answer the question by returing A or B.\n\n'

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem

        e0 = data['e0']
        e1 = data['e1']
        choices = [e0, e1]

        src, lab = self.wrap_data(data)
        prompt += src

        srcs = [prompt]

        yield srcs, [choices], lab

    def re_MCTACO(self, data):

        prompt = 'Answer the question by returing A or B.\n\n'

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem

        e0 = data['e0']
        e1 = data['e1']
        choices = ['before', 'after']

        src, lab = self.wrap_data(data)
        prompt += src

        srcs = [prompt]

        yield srcs, [choices], lab

    def mc_COAP(self, data):

        # vicuna
        # prompt = 'Answer the question by returning A or B.\n\n'
        # prompt = ''

        prompt = 'Answer the question by selecting A, B\n\n'
        # prompt = 'Answer the question using only the letter label of the option.\n\n'

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem

        e0 = data['e0']
        e1 = data['e1']
        choices = [e0, e1]

        src, lab = self.wrap_data(data)
        prompt += src

        srcs = [prompt]

        yield srcs, [choices], lab

    def mc_MCTACO(self, data):

        # vicuna
        # prompt = 'Answer the question by returning A or B.\n\n'
        # prompt = ''

        prompt = 'Answer the question by selecting A, B\n\n'
        # prompt = 'Answer the question using only the letter label of the option.\n\n'

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem

        choices = data['choices']

        src, lab = self.wrap_data(data)
        prompt += src

        srcs = [prompt]

        yield srcs, [['A', 'B']], lab

    def gen_MCTACO(self, data):

        # vicuna
        # prompt = 'Answer the question by returning A or B.\n\n'
        prompt = ''

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem

        choices = [data['lab']]

        src, lab = self.wrap_data(data)
        prompt += src

        srcs = [prompt]

        yield srcs, [choices], lab

    def nli_MCTACO(self, data):

        prompt = 'Answer the question by returning A or B.\n\n'

        # prompt = 'Answer the question by returing True or False.\n\n'

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem

        e0 = data['e0']
        e1 = data['e1']
        choices = [e0, e1]

        src, lab = self.wrap_data(data)
        prompt += src

        srcs = [prompt]

        yield srcs, [choices], lab
    
    def nli_MCTACO_v2(self, data):

        prompt = 'Answer the question by returning A or B.\n\n'

        # prompt = 'Decide whether the following question\'s answer is correct by returing A or B.\n\n'

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem
        context=data['context']
        question=data['question']
        if question.endswith('?'):
            question = question[:-1]
        e1 = data['e1'].capitalize()
        # labls={false:"False"}
        lab2ind = {0 : 'A', 1 : 'B'}
        lab=lab2ind[data['lab']]
        # choices = [e1]
        src = f'Context:\n{context}\nQuestion:\nIs \"{e1}\" the answer to \"{question}\"?\nChoices:\nA. False\nB. True\nAnswer:\n'
        #src, lab = self.wrap_data(data)
        # src = src.replace('<ctx>', context).replace('<q>', question).replace('<e>',e1)
        prompt += src

        srcs = [prompt]

        yield srcs, ['A', 'B'], lab
        
    # def nli_MCTACO_v2(self, data):

    #     # prompt = 'Answer the question by returning A or B.\n\n'

    #     prompt = 'Decide whether the following question\'s answer is correct by returing True or False.\n\n'

    #     dem = self.build_demonstrations(list(range(self.k)))
    #     prompt += dem
    #     context=data['context']
    #     question=data['question']
    #     if not question.endswith('?'):
    #         question += '?'
    #     e1 = data['e1'].capitalize()
    #     # labls={false:"False"}
    #     lab2ind = {0 : 'f', 1 : 't'}
    #     lab=lab2ind[data['lab']]
    #     choices = [e1]
    #     src = 'Context:\n<ctx>\nQuestion:\n<q>\nAnswer:\n<e>.'
    #     #src, lab = self.wrap_data(data)
    #     src = src.replace('<ctx>', context).replace('<q>', question).replace('<e>',e1)
    #     prompt += src

    #     srcs = [prompt]

    #     yield srcs, [choices], lab

    def mc_STC(self, data):

        # prompt = 'Answer the question by selecting A, B\n\n'
        # vicuna
        prompt = 'Answer the question by returning A or B\n\n'

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem

        e0 = data['e0']
        e1 = data['e1']
        choices = [e0, e1]

        src, lab = self.wrap_data(data)
        prompt += src

        srcs = [prompt]

        yield srcs, [['A', 'B']], lab

    def mc_SCITE(self, data):

        # vicuna
        # prompt = 'Answer the question by returning A or B.\n\n'
        # prompt = ''

        prompt = 'Answer the question by selecting A, B\n\n'
        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem

        choices = ['no relation', 'causal']

        src, lab = self.wrap_data(data)
        prompt += src

        srcs = [prompt]

        yield srcs, [choices], lab

    def mc_MCNC(self, data):

        choices = data['choices']

        # prompt = 'Answer the question by selecting A, B, C, D, E. Only return the alphabetical number of the correct answer.\n\n'
        # vicuna
        prompt = 'Answer the question by returning A, B, C, D or E. Only return the alphabetical number of the correct answer.\n\n'

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem

        src, lab = self.wrap_data(data)
        prompt += src

        srcs = [prompt]

        yield srcs, [['A', 'B', 'C', 'D', 'E']], lab

    def gen_MCNC(self, data):

        choices = data['choices']

        prompt = ''

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem

        src, lab = self.wrap_data(data)
        prompt += src

        srcs = [prompt]

        yield srcs, [choices], lab

    def mc_MCNC_json(self, data):

        lab2ind = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

        def wrap_event_json(event):
            event['action'] = event.pop('predicate')
            event = {k: v for k, v in event.items() if v is not None}
            e_str = str(event)
            return e_str

        events = [wrap_event_json(e) for e in data['events']]
        choices = [wrap_event_json(e) for e in data['choices']]
        lab = data['lab']

        doc = ',\n'.join(events)
        # src = f'Answer the question by selecting A, B, C, D, E. Only return the alphabetical number of the correct answer.\nEvents:\n{doc}\nQuestion:\nWhat is the next event?\nChoices:\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nE. {choices[4]}\nAnswer:\n'
        src = f'Answer the question by only returning the alphabetical number of the correct answer without explanation.\nGiven a list of events sorted by time and represented in json format:\n[{doc}]\nQuestion:\nWhich event would be most appropriate to add to this list?\nChoices:\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nE. {choices[4]}\nAnswer:\n'

        srcs = [src]

        yield srcs, [choices], lab2ind[lab]

    def gen_MCNC_json(self, data):

        choices = data['choices']

        prompt = 'Given a sequence of events as context, answer the question by return a JSON object with subject, action, object, preposition as keys.\n'

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem

        src, lab = self.wrap_data(data)
        prompt += src

        srcs = [prompt]

        yield srcs, [choices], lab



    def lmmc_DTFit(self, data):

        lab2ind = {0: 'A', 1: 'B'}

        doc0 = data['doc0']
        doc1 = data['doc1']
        lab = data['lab']

        docs = [doc0.lower(), doc1.lower()]
        
        srcs = []
        for doc in docs:
            src_ = doc
            srcs.append(src_)

        yield srcs, docs, lab2ind[lab]


    def lmmc_HardSimExt(self, data):

        lab2ind = {0: 'A', 1: 'B'}

        doc = data['doc']
        e0 = data['e0']
        e1 = data['e1']
        lab = data['lab']

        prompt = f'\"{doc} .\" is semantically similar to \"<event> .\".' 
        es = [e0.lower(), e1.lower()]
        
        srcs = []
        for e in es:
            src_ = prompt.replace('<event>', e)
            srcs.append(src_)

        yield srcs, [es], lab2ind[lab]

    def mc_HardSimExt(self, data):

        e0 = data['e0']
        e1 = data['e1']
        choices = [e0, e1]

        # prompt = 'Answer the question by selecting A, B\n\n'

        # vicuna
        prompt = 'Answer the question by returning A or B.\n\n'

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem

        src, lab = self.wrap_data(data)
        prompt += src

        srcs = [prompt]

        yield srcs, [choices], lab

    def mc_HardSimExt_json(self, data):

        lab2ind = {0: 'A', 1: 'B'}

        def wrap_event_json(event):
            event = {k: v for k, v in event.items() if v is not None}
            e_str = str(event)
            return e_str

        e = data['e']
        e0 = wrap_event_json(data['e0'])
        e1 = wrap_event_json(data['e1'])
        lab = data['lab']

        prompt = f'Answer the question by only returning the alphabetical number of the correct answer without explanation.\nQuestion:\nWhich event is semantically similar to {e}\nChoices:\nA. <e0>\nB. <e1>\nAnswer:\n' 
        es = [e0, e1]

        src = prompt.replace('<e0>', e0).replace('<e1>', e1) 

        srcs = [src]

        yield srcs, [es], lab2ind[lab]


    def mc_SocialIQA(self, data):

        # prompt = 'Answer the question by selecting A, B or C.\n'
        # vicuna
        prompt = 'Answer the question by returning A, B or C.\n\n'

        # prompt = 'Answer the question using only the letter label of the option.\n\n'

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem

        choices = data['choices']

        src, lab = self.wrap_data(data)
        prompt += src

        srcs = [prompt]

        yield srcs, [['A', 'B', 'C']], lab

    def re_SocialIQA(self, data):

        # prompt = 'Answer the question by selecting A, B or C.\n'
        # vicuna
        prompt = 'Answer the question by returning A, B.\n\n'

        # prompt = 'Answer the question using only the letter label of the option.\n\n'

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem

        choices = data['choices']

        src, lab = self.wrap_data(data)
        prompt += src

        srcs = [prompt]

        yield srcs, [['A', 'B']], lab



    def nli_SocialIQA(self, data):


        prompt = 'Answer the question by returing True or False.\n\n'
        src, lab = self.wrap_data(data)

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem
        prompt += src
        srcs = [prompt]

        choices = ['True', 'False']

        yield srcs, [choices], lab

    def gen_SocialIQA(self, data):

        # prompt = 'Answer the question by selecting A, B or C.\n'
        # vicuna

        prompt = ''
        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem

        choices = data['choices']

        src, lab = self.wrap_data(data)
        prompt += src

        srcs = [prompt]

        yield srcs, [choices], lab


    def nli_TRACIE(self, data):

        # prompt = 'Answer the question by selecting A, B\n\n'

        # # vicuna
        # prompt = 'Answer the question by returning A or B.\n\n'

        # # ESP_v1
        # prompt = 'Is the following statement true? Answer the question by returning Entailment or Contradiction.\n\n'

        # ESP_v1.2
        prompt = 'Answer the question by returning A or B.\n\n'

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem

        src, lab = self.wrap_data(data)
        prompt += src

        srcs = [prompt]
        # choices = ['Entailment', 'Contradiction']
        choices = ['False', 'True']

        yield srcs, [choices], lab

    def nli_ALTLEX(self, data):

        # prompt = 'Answer the question by selecting A, B\n\n'
        # # vicuna
        # prompt = 'Answer the question by returning A or B.\n\n'
        # vicuna
        prompt = 'Answer the question by returning True or False.\n\n'

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem

        src, lab = self.wrap_data(data)
        prompt += src

        srcs = [prompt]

        yield srcs, [['False', 'True']], lab

    def gen_CQA(self, data):

        def build_demonstrations(doc_indices):

            seq = ''
            for ind in doc_indices:
                state, _ = self.wrap_data(self.trainset[ind], add_lab=True)

                seq += state
                seq += '\n\n' 

            return seq

        src, tgt = self.wrap_data(data)

        # demonstrations = build_demonstrations(list(range(self.k)))
        # src = 'Answer the question by using sentences from the article.\n\n' + demonstrations + src

        srcs = [src]

        yield srcs, [tgt], tgt
    def gen_CQA_v2(self,data):
        prompt = ''

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem
        choices=[data['lab']]
        # we donnot care about choices
        lab=data['lab']
        context=data['context']
        question=data['question']
        src = f'Context:\n{context}\nQuestion:\n{question}\nAnswer:'
        prompt += src
        srcs=[prompt]
        yield srcs, [choices], lab
    def mc_ESTER(self, data):

        prompt = 'Answer the question by selecting A, B, C, D\n\n'
        # prompt = 'Answer the question using only the letter label of the option.\n\n'

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem

        choices = data['choices']

        src, lab = self.wrap_data(data)
        prompt += src

        srcs = [prompt]

        yield srcs, [['A', 'B', 'C', 'D']], lab

    def mc_explicit(self, data):

        prompt = 'Answer the question by selecting A, B, C, D\n\n'
        # prompt = 'Answer the question using only the letter label of the option.\n\n'

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem
        lab2ind = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        choices = data['choices']
        context = data['context']
        question = data['question']
        lab=lab2ind[choices.index(data['e1'])]
        inds = [char for char in string.ascii_letters if char.isupper()]
        ipt = ''.join([f'{inds[i]}. {c}\n' for i, c in enumerate(choices)])

        # unified prompt
        src = f'Context:\n{context}\nQuestion:\n{question}\nChoices:\n{ipt}\nThe answer is'
        # src, lab = self.wrap_data(data)
        prompt += src

        srcs = [prompt]

        yield srcs, [['A', 'B', 'C', 'D']], lab

    def gen_ESTER(self, data):
        prompt = ''

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem

        choices = data['choices']

        src, lab = self.wrap_data(data)
        prompt += src

        srcs = [prompt]

        yield srcs, [choices], lab

    def gen_ESTER_v2(self,data):
        prompt = ''

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem
        choices=[data['lab']]
        # we donnot care about choices
        lab=data['lab']
        context=data['context']
        question=data['question']
        src = f'Context:\n{context}\nQuestion:\n{question}\nAnswer:'
        prompt += src
        srcs=[prompt]
        yield srcs, [choices], lab

    def re_ESTER(self, data):

        # prompt = 'Answer the question by selecting A, B or C.\n'
        # vicuna
        prompt = 'Answer the question by returning A, B.\n\n'

        # prompt = 'Answer the question using only the letter label of the option.\n\n'

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem

        choices = data['choices']

        src, lab = self.wrap_data(data)
        prompt += src

        srcs = [prompt]

        yield srcs, [['A', 'B']], lab
    
    def re_MATRES(self, data):

        # prompt = 'Answer the question by selecting A, B or C.\n'
        # vicuna
        prompt = 'Determine the type of temporal relationship between events by returning A, B, C or D.\n\n'

        # prompt = 'Answer the question using only the letter label of the option.\n\n'

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem

        # choices = data['choices']
        lab_map={"BEFORE":"A","SIMULTANEOUS":"B","AFTER":"C","VAGUE":"D"}
        doc= capitalize_sentences(data['doc'])
        e0=data['e0']
        e1=data['e1']
        lab=lab_map[data['lab']]
        # src, lab = self.wrap_data(data)
        src=f'Context:\n{doc}\nQuestion:\nWhat is the temporal relationship between "{e0}" and "{e1}"?\nChoices:\nA. "{e0}" happened before "{e1}".\nB. "{e0}" and "{e1}" happened simultaneously.\nC. "{e0}" happended after "{e1}".\nD. Can\'t decide.\nAnswer:\n'
        prompt += src

        srcs = [prompt]

        yield srcs, [['BEFORE', 'SIMULTANEOUS', 'AFTER', 'VAGUE']], lab

    # def re_MATRES(self, data):

    #     # prompt = 'Answer the question by selecting A, B or C.\n'
    #     # vicuna
    #     prompt = 'Determine the type of temporal relationship between events by returning A, B, C or D.\n\n'

    #     # prompt = 'Answer the question using only the letter label of the option.\n\n'

    #     dem = self.build_demonstrations(list(range(self.k)))
    #     prompt += dem

    #     # choices = data['choices']
    #     lab_map={"BEFORE":"A","SIMULTANEOUS":"B","AFTER":"C","VAGUE":"D"}
    #     doc= capitalize_sentences(data['doc'])
    #     e0=data['e0']
    #     e1=data['e1']
    #     lab=lab_map[data['lab']]
    #     # src, lab = self.wrap_data(data)
    #     # src=f'Context:\n{doc}\nQuestion:\nIs "{e0}" before, simultaneous, after or vague "{e1}"?\nChoices:\nA. Before\nB. Simultaneous\nC. After\nD. Vague\nThe answer is'
    #     src=f'Context:\n{doc}\nQuestion:\nWhat is the temporal relation between "{e0}" and "{e1}"?\nChoices:\nA. "{e0}" Before "{e1}".\nB. "{e0}" and "{e1}" happened Simultaneously.\nC. "{e0}" After "{e1}".\nD. Can not decide.\nThe answer is'
    #     prompt += src

    #     srcs = [prompt]

    #     yield srcs, [['BEFORE', 'SIMULTANEOUS', 'AFTER', 'VAGUE']], lab
    
    def re_ESL_D(self, data):

        # prompt = 'Answer the question by selecting A, B or C.\n'
        # vicuna
        prompt = 'Determine the type of causal relationship between events by returning A, B or C.\n\n'

        # prompt = 'Answer the question using only the letter label of the option.\n\n'

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem

        # choices = data['choices']
        lab_map={"CAUSE":"A","EFFECT":"B","None":"C"}
        doc= (data['doc'])
        e0=data['e0']
        e1=data['e1']
        lab=lab_map[data['lab']]
        # src, lab = self.wrap_data(data)
        src=f'Context:\n{doc}\nQuestion:\nWhat is the causal relation between "{e0}" and "{e1}"?\nChoices:\nA. "{e1}" caused "{e0}".\nB. "{e0}" caused "{e1}".\nC. There is no causal relationship between them.\nAnswer:\n'
        prompt += src

        srcs = [prompt]

        yield srcs, [['CAUSE', 'EFFECT', 'None']], lab
    
    # def re_ESL_D(self, data):

    #     # prompt = 'Answer the question by selecting A, B or C.\n'
    #     # vicuna
    #     prompt = 'Determine the type of causal relationship between events by returning A, B or C.\n\n'

    #     # prompt = 'Answer the question using only the letter label of the option.\n\n'

    #     dem = self.build_demonstrations(list(range(self.k)))
    #     prompt += dem

    #     # choices = data['choices']
    #     lab_map={"CAUSE":"A","EFFECT":"B","None":"C"}
    #     doc= (data['doc'])
    #     e0=data['e0']
    #     e1=data['e1']
    #     lab=lab_map[data['lab']]
    #     # src, lab = self.wrap_data(data)
    #     src=f'Context:\n{doc}\nQuestion:\nWhat is the causal relation from "{e0}" to "{e1}"?\nChoices:\nA. cause.\nB. result.\nC. None.\nThe answer is'
    #     prompt += src

    #     srcs = [prompt]

    #     yield srcs, [['CAUSE', 'EFFECT', 'None']], lab
        
    def re_explicit(self, data):

        prompt = 'Answer the question by selecting A, B or C.\n\n'
        # vicuna
        # prompt = 'Determine the type of causal relationship between events by returning A, B or C.\n\n'

        # prompt = 'Answer the question using only the letter label of the option.\n\n'

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem

        # choices = data['choices']
        lab_map={"HasSubevent":"A", "IsSubevent":"B","Before":"A",  "After":"B","Causes":"A", "IsResult":"B","Vague" : "C"}
        choices = data['choices']
        context = data['context']
        question = data['question']
        inds = [char for char in string.ascii_letters if char.isupper()]
        ipt = ''.join([f'{inds[i]}. {c}\n' for i, c in enumerate(choices)])
        
        lab=lab_map[data['re1']]
        # src, lab = self.wrap_data(data)
        src=f'Context:\n{context}\nQuestion:\n{question}\nChoices:\n{ipt}\nThe answer is'
        prompt += src

        srcs = [prompt]

        yield srcs, [['A', 'B', 'C']], lab


    def nli_ESTER(self, data):


        prompt = 'Answer the question by returing True or False.\n\n'
        src, lab = self.wrap_data(data)

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem
        prompt += src
        srcs = [prompt]

        choices = ['True', 'False']

        yield srcs, [choices], lab

    def gen_SE2020T5_EQA(self, data):

        def build_demonstrations(doc_indices):

            seq = ''
            for ind in doc_indices:
                state, _ = self.wrap_data(self.trainset[ind], add_lab=True)

                seq += state
                seq += '\n\n' 

            return seq

        src, tgt = self.wrap_data(data)

        # demonstrations = build_demonstrations(list(range(self.k)))
        # src = 'Answer the question by using sentences from the article.\n\n' + demonstrations + src

        srcs = [src]

        yield srcs, [tgt], tgt


    def gen_StoryGen(self, data):

        src, tgt = self.wrap_data(data)

        demonstrations = self.build_demonstrations(list(range(self.k)))
        if demonstrations:
            src = demonstrations + src

        srcs = [src]

        yield srcs, [tgt], tgt

    def mc_TIMETRAVEL(self, data):

        lab2ind = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

        prompt = 'Answer the question by returing A or B.\n\n'
        src, lab = self.wrap_data(data)

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem
        prompt += src
        srcs = [prompt]

        original_ending = data['original_ending']
        tgt = data['lab'][0]

        if lab == 0:
            choices = [tgt, original_ending]
        else:
            choices = [original_ending, tgt]

        yield srcs, [choices], lab2ind[lab]

    def nli_TIMETRAVEL(self, data):


        prompt = 'Answer the question by returing True or False.\n\n'
        src, lab = self.wrap_data(data)

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem
        prompt += src
        srcs = [prompt]

        choices = [True, False]

        yield srcs, [choices], lab

    def gen_TIMETRAVEL(self, data):

        prompt = ''
        src, tgt = self.wrap_data(data)

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem
        prompt += src
        srcs = [prompt]

        yield srcs, [tgt], tgt

    def __getitem__(self, index):
        src_list = self.src_list[index]
        stm_list = self.stm_list[index]
        lab = self.labs[index]
        return src_list, stm_list, lab 


    def __len__(self,):
        return len(self.src_list)

        
    def collect_fn(self, data):
        src = list(zip(*[d[0] for d in data]))
        stm = list(zip(*[d[1] for d in data]))
        labs = [d[2] for d in data]

        return {
                'src': src,
                'stm': stm,
                'labs': labs,
               }
