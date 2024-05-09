import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import json
import re
from collections import Counter
import spacy

with open('/data1/tzw/data/explicit/clustering.json') as f: et_clusters = json.load(f)
et2i = {}
for i, ets in enumerate(et_clusters):
    for et in ets:
        et2i[et] = i

def compute_unigram_f1(pred_counter, gold_counter):
    corr = 0
    for w in pred_counter:
        if w in gold_counter:
            corr += min(pred_counter[w], gold_counter[w])

    prec = float(corr) / sum(pred_counter.values()
                             ) if sum(pred_counter.values()) > 0 else 0.0
    recl = float(corr) / sum(gold_counter.values()
                             ) if sum(gold_counter.values()) > 0 else 0.0
    return recl, prec, 2 * prec * recl / (prec + recl) if prec + recl > 0 else 0.0

def simpson_score(text1, text2):
    event1_tokens = set(text1.split())
    event2_tokens = set(text2.split())
    try:
        Simpson = len(event1_tokens & event2_tokens) / min(len(event1_tokens), len(event2_tokens))
        return Simpson
    except ZeroDivisionError:
        return 0

def extract_json_dict(json_string):
    

    json_match = re.search(r'```json\s*(.+)\s*```', json_string, re.DOTALL)

    if json_match:
        json_string = json_match.group(1)
        
    json_dict = None

    try:
        json_dict = json.loads(json_string)
    except:
        pass
    return json_dict

class LMMetrics(object):
    def __init__(self, generative_format, task_name, use_cot=False):
        self.generative_format = generative_format
        self.task_name = task_name
        self.use_cot = use_cot

        self.compute_process = {
                'lmmc': self.metrics_lmmc,
                'mc': self.metrics_mc_v1,
                're': self.metrics_mc_v1,
                'nli': self.metrics_nli,
                'gen': self.metrics_gen,
                'eae': self.metrics_eae,
                }
        self.compute = self.compute_process[generative_format]

    def the_answer_is(self, text):
        pattern = r"the answer is [A-Z]"
        match = re.search(pattern, text)

        if match:
            answer = match.group(0)
            return answer
        else:
            return 'A'

    def unwrap_event_json(self, event):
        try:
            event = json.loads(event)
        except:
            return 'none'
        res = ''
        if 'subject' in event:
            res += event['subject']
            res += ' '
        if 'action' in event:
            res += event['action']
            res += ' '
        if 'object' in event:
            res += event['object']
            res += ' '
        if 'preposition' in event:
            res += event['preposition']
            res += ' '

        res = res.strip()
        return res

    def compute_result(self, preds, tgts, labs):
        return self.compute(preds, tgts, labs)

    def metrics_lmmc(self, preds, tgts, labs):
        ind2lab = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        gold_ids = []
        pred_ids = []

        for pred, lab in zip(preds, labs):
            pred = np.array(pred)
            pid = pred.argmax()
            gold_ids.append(ind2lab[lab])
            pred_ids.append(pid)
        
        acc = accuracy_score(np.array(gold_ids, dtype=np.int64), 
                      np.array(pred_ids, dtype=np.int64))

        results = {}
        results['acc'] = 100.0 * round(acc, 4)
        return results

    def metrics_mc(self, preds, tgts, labs):

        ind2lab = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        gold_ids = []
        pred_ids = []

        for pred, tgt, lab in zip(preds, tgts, labs):
            if pred:
                pred = pred.strip()[0]

            gold_ids.append(ind2lab[lab])
            if pred not in tgt:
                pred_ids.append(0)
            else:
                pred_ids.append(ind2lab[pred])

        
        acc = accuracy_score(np.array(gold_ids, dtype=np.int64), 
                      np.array(pred_ids, dtype=np.int64))

        results = {}
        results['acc'] = 100.0 * round(acc, 4)
        return results

    def metrics_mc_v1(self, preds, choices, labs):

        lab2ind = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7}

        gold_ids = []
        pred_ids = []
        for pred, choice, lab in zip(preds, choices, labs):
            if pred:
                pred = pred.strip()
            ind = lab2ind[lab]
            gold_ids.append(ind)

            if any([pred.startswith(f'{k}.') for k in lab2ind.keys()]):
                pid = [pred.startswith(f'{k}.') for k in lab2ind.keys()].index(True)
                pred_ids.append(pid)
            elif re.search(r'[A-Z]\.', pred):
                m = re.search(r'[A-Z]\.', pred)
                pred = m.group(0)[0]
                pred_ids.append(lab2ind.get(pred, 0))
            elif re.search(r'[A-Z],', pred):
                m = re.search(r'[A-Z],', pred)
                pred = m.group(0)[0]
                pred_ids.append(lab2ind.get(pred, 0))
            elif re.search(r'\([A-Z]\)', pred):
                m = re.search(r'\([A-Z]\)', pred)
                pred = m.group(0)[1]
                pred_ids.append(lab2ind.get(pred, 0))
            elif any([pred == k for k in lab2ind.keys()]):
                pid = [pred == k for k in lab2ind.keys()].index(True)
                pred_ids.append(pid)
            else:
                p0 = r"the(?: correct)? answer is[\s:]+([ABCDEFGH])"
                m0 = re.search(p0, pred, re.IGNORECASE)
                p1 = r"the(?: correct)? (?:option|answer) should be[\s:]+([ABCDEFGH])"
                m1 = re.search(p1, pred, re.IGNORECASE)

                if m0:
                    pred = m0.group(1)[-1].capitalize()
                    pred_ids.append(lab2ind.get(pred, 0))
                elif m1:
                    pred = m1.group(1)[-1].capitalize()
                    pred_ids.append(lab2ind.get(pred, 0))
                else:
                    print(pred)
                    simpsom_scores = [simpson_score(c, pred) for c in choice] 
                    pred_ids.append(np.argmax(simpsom_scores))
                    print(np.argmax(simpsom_scores))

        
        acc = accuracy_score(np.array(gold_ids, dtype=np.int64), 
                      np.array(pred_ids, dtype=np.int64))
        
        print(acc)

        results = {}
        results['acc'] = 100.0 * round(acc, 4)
        return results

    def metrics_eae(self, preds, tgts, labs):

        ind2lab = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

        gold_ids = []
        pred_ids = []

        for pred, tgt, lab in zip(preds, tgts, labs):

            if pred:
                pred = pred.strip().lower()

            lab = lab.lower()

            if lab in pred:
                pred_ids.append(1)
            else:
                pred_ids.append(0)

            gold_ids.append(1)

        acc = accuracy_score(np.array(gold_ids, dtype=np.int64), 
                      np.array(pred_ids, dtype=np.int64))

        results = {}
        results['acc'] = 100.0 * round(acc, 4)
        return results

    def metrics_re(self, preds, tgts, labs):

        ind2lab = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

        gold_ids = []
        pred_ids = []

        for pred, tgt, lab in zip(preds, tgts, labs):

            if pred:
                pred = pred.strip()[0]

            gold_ids.append(ind2lab[lab])
            pred_ids.append(ind2lab.get(pred, 0))

        
        acc = accuracy_score(np.array(gold_ids, dtype=np.int64), 
                      np.array(pred_ids, dtype=np.int64))

        results = {}
        results['acc'] = 100.0 * round(acc, 4)
        return results

    def metrics_nli(self, preds, tgts, labs):
        ind2lab = {"A": 0, "B": 1}

        gold_ids = []
        pred_ids = []

        for pred, tgt, lab in zip(preds, tgts, labs):

            if pred:
                pred = pred.strip()

            gold_ids.append(ind2lab[lab[0]])
            if not pred:
                pred_ids.append(0)
            else:
                mt = re.search(r'[Tt]rue', pred)
                mf = re.search(r'[Ff]alse', pred)
                if mt:
                    pred_ids.append(1)
                elif mf:
                    pred_ids.append(0)
                else:
                    print(pred)
                    pred_ids.append(0)

        acc = accuracy_score(np.array(gold_ids, dtype=np.int64), 
                      np.array(pred_ids, dtype=np.int64))

        f1 = f1_score(np.array(gold_ids, dtype=np.int64), 
                      np.array(pred_ids, dtype=np.int64))
        results = {}
        results['acc'] = 100.0 * round(acc, 4)
        results['f1'] = 100.0 * round(f1, 4)
        return results

    def metrics_gen(self, preds, tgts, labs):

        eval_f1 = []
        nlp = spacy.load('en_core_web_lg')
        refs = []
        for pred, gold in zip(preds, labs):
            if not pred:
                pred = 'None'

            if 'json' in self.task_name:
                pred = self.unwrap_event_json(pred)
                if isinstance(gold, str):
                    gold = self.unwrap_event_json(gold)
                else:
                    gold = [self.unwrap_event_json(g) for g in gold]

            pred_tokens = [pred.text for pred in nlp(pred)]

            if isinstance(gold, str):
                gold_tokens = [[gold.text for gold in nlp(gold)]]
                refs.append([gold])
            else:
                gold_tokens = [[t.text for t in nlp(g)] for g in gold]
                refs.append(gold) # TODO

            # F1
            f1 = 0
            for gtokens in gold_tokens:
                # construct unigram counter
                pred_counter = Counter(pred_tokens)
                gold_counter = Counter(gtokens)
                _, _, f1_ = compute_unigram_f1(pred_counter, gold_counter)
                f1 += f1_

            f1 /= len(gold_tokens)

            eval_f1.append(f1)

        F1 = round(np.mean(eval_f1), 4) * 100

        results = {}
        results['f1'] = F1

        return results
    

    def metrics_rq_num(self, preds, choices, labs):
        import json
        from collections import Counter

        def calculate_accuracy_and_f1(pred_dict, lab_dict):
            
            correct = sum(key in lab_dict['event_type'] and pred_dict['event_type'][key] == lab_dict['event_type'][key] 
                        for key in pred_dict['event_type'])
            total = len(pred_dict['event_type'])
            accuracy = correct / total if total > 0 else 0
            
            pd = []
            for t in pred_dict['event_relation']:
                if len(t) != 3:
                    pd.append([-1, 'Causes', -1])
                else:
                    pd.append([et2i.get(t[0], -1), t[1], et2i.get(t[2], -1)])
            
            pred_dict['event_relation'] = pd
            lab_dict['event_relation'] = [[et2i.get(t[0], -1), t[1], et2i.get(t[2], -1)] for t in lab_dict['event_relation']]

            pred_relations = Counter(tuple(relation) for relation in pred_dict['event_relation'])
            lab_relations = Counter(tuple(relation) for relation in lab_dict['event_relation'])
            
            common_relations = pred_relations & lab_relations
            tp = sum(common_relations.values())  # 真正例 (True Positives)
            fp = sum((pred_relations - lab_relations).values())  # 假正例 (False Positives)
            fn = sum((lab_relations - pred_relations).values())  # 假负例 (False Negatives)

            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

            return accuracy, f1

        def process_lists(preds, labs):
            for pred,lab in zip(preds,labs):
                print(pred)
                print(lab)
            preds_dicts = []
            for pred in preds:
                try:
                    preds_dicts.append(eval(pred))
                    assert isinstance(eval(pred), dict)
                except:
                    preds_dicts.append({"event_type":{}, "event_relation":[]})

            labs_dicts = [eval(lab) for lab in labs]

            total_event_type_elements = sum(len(pred_dict['event_type']) for pred_dict in preds_dicts)
            total_event_relation_elements = sum(len(pred_dict['event_relation']) for pred_dict in preds_dicts)

            weighted_accuracy_sum = 0
            weighted_f1_sum = 0

            for pred_dict, lab_dict in zip(preds_dicts, labs_dicts):
                # print(pred_dict)
                # print(lab_dict)
                if pred_dict == {"event_type":{}, "event_relation":[]}:
                    continue
                accuracy, f1 = calculate_accuracy_and_f1(pred_dict, lab_dict)

                weight_event_type = len(pred_dict['event_type']) / total_event_type_elements if total_event_type_elements > 0 else 0
                weight_event_relation = len(pred_dict['event_relation']) / total_event_relation_elements if total_event_relation_elements > 0 else 0

                weighted_accuracy_sum += accuracy * weight_event_type
                weighted_f1_sum += f1 * weight_event_relation

            return weighted_accuracy_sum, weighted_f1_sum

        results={}
        results['acc'], results['F1'] =process_lists(preds, labs)
        
        return results
    
    def metrics_rq3(self, preds, choices, labs):

        def calculate_accuracy_and_f1(pred_dict, lab_dict):
            correct = sum(pred_dict['event_type'][key] == lab_dict['event_type'][key] 
                        for key in pred_dict['event_type'])
            total = len(pred_dict['event_type'])
            accuracy = correct / total if total > 0 else 0

            pred_relations = Counter(tuple(relation) for relation in pred_dict['event_relation'])
            lab_relations = Counter(tuple(relation) for relation in lab_dict['event_relation'])

            common_relations = pred_relations & lab_relations
            tp = sum(common_relations.values())  # 真正例 (True Positives)
            fp = sum((pred_relations - lab_relations).values())  # 假正例 (False Positives)
            fn = sum((lab_relations - pred_relations).values())  # 假负例 (False Negatives)

            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

            return accuracy, f1

        def process_lists(preds, labs):
            for pred,lab in zip(preds,labs):
                print(pred)
                print(lab)
            preds_dicts = []
            for pred in preds:
                try:
                    preds_dicts.append(eval(pred))
                except:
                    preds_dicts.append({"event_type":{}, "event_relation":[]})

            labs_dicts = [eval(lab) for lab in labs]

            total_event_type_elements = sum(len(pred_dict['event_type']) for pred_dict in preds_dicts)
            total_event_relation_elements = sum(len(pred_dict['event_relation']) for pred_dict in preds_dicts)

            weighted_accuracy_sum = 0
            weighted_f1_sum = 0

            for pred_dict, lab_dict in zip(preds_dicts, labs_dicts):
                if pred_dict == {"event_type":{}, "event_relation":[]}:
                    continue
                accuracy, f1 = calculate_accuracy_and_f1(pred_dict, lab_dict)

                weight_event_type = len(pred_dict['event_type']) / total_event_type_elements if total_event_type_elements > 0 else 0
                weight_event_relation = len(pred_dict['event_relation']) / total_event_relation_elements if total_event_relation_elements > 0 else 0

                weighted_accuracy_sum += accuracy * weight_event_type
                weighted_f1_sum += f1 * weight_event_relation

            return weighted_accuracy_sum, weighted_f1_sum

        results={}
        results['acc'], results['F1'] =process_lists(preds, labs)
        
        return results

    def metrics_rq_num(self, preds, choices, labs):
        import json
        from collections import Counter

        def calculate_accuracy_and_f1(pred_dict, lab_dict):
            
            correct = sum(key in lab_dict['event_type'] and pred_dict['event_type'][key] == lab_dict['event_type'][key] 
                        for key in pred_dict['event_type'])
            total = len(pred_dict['event_type'])
            accuracy = correct / total if total > 0 else 0

            pd = []
            for t in pred_dict['event_relation']:
                if len(t) != 3:
                    pd.append([-1, 'Causes', -1])
                else:
                    pd.append([et2i.get(t[0], -1), t[1], et2i.get(t[2], -1)])
            
            pred_dict['event_relation'] = pd
            lab_dict['event_relation'] = [[et2i.get(t[0], -1), t[1], et2i.get(t[2], -1)] for t in lab_dict['event_relation']]

            pred_relations = Counter(tuple(relation) for relation in pred_dict['event_relation'])
            lab_relations = Counter(tuple(relation) for relation in lab_dict['event_relation'])
            
            common_relations = pred_relations & lab_relations
            tp = sum(common_relations.values())  # 真正例 (True Positives)
            fp = sum((pred_relations - lab_relations).values())  # 假正例 (False Positives)
            fn = sum((lab_relations - pred_relations).values())  # 假负例 (False Negatives)

            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

            return accuracy, f1

        def process_lists(preds, labs):
            preds_dicts = []
            for pred in preds:
                p = extract_json_dict(pred)
                if p is None:
                    preds_dicts.append({"event_type":{}, "event_relation":[]})
                else:
                    print(pred)
                    preds_dicts.append(p)
                    assert isinstance(extract_json_dict(pred), dict)
                    

            labs_dicts = [extract_json_dict(lab) for lab in labs]

            total_event_type_elements = sum(len(pred_dict['event_type']) for pred_dict in preds_dicts)
            total_event_relation_elements = sum(len(pred_dict['event_relation']) for pred_dict in preds_dicts)

            weighted_accuracy_sum = 0
            weighted_f1_sum = 0

            for pred_dict, lab_dict in zip(preds_dicts, labs_dicts):
                if pred_dict == {"event_type":{}, "event_relation":[]}:
                    continue
                accuracy, f1 = calculate_accuracy_and_f1(pred_dict, lab_dict)

                weight_event_type = len(pred_dict['event_type']) / total_event_type_elements if total_event_type_elements > 0 else 0
                weight_event_relation = len(pred_dict['event_relation']) / total_event_relation_elements if total_event_relation_elements > 0 else 0

                weighted_accuracy_sum += accuracy * weight_event_type
                weighted_f1_sum += f1 * weight_event_relation

            return weighted_accuracy_sum, weighted_f1_sum

        results={}
        results['acc'], results['F1'] =process_lists(preds, labs)

        return results

