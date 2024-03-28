import json
import os
from torch.utils.data import Dataset
import logging
import random

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

class LlamaSTFDataset(Dataset):
    def __init__(self, file, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.max_seq_length = max_seq_length
        logging.info('Loading data: {}'.format(file))
        with open(file, 'r', encoding='utf8') as f:
            # data_list = json.load(f)
            data_list = f.readlines()

        random.shuffle(data_list)
        
        logging.info("There are {} data in dataset".format(len(data_list)))
        self.data_list = data_list
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        prompt_input, prompt_no_input = PROMPT_DICT['prompt_input'], PROMPT_DICT['prompt_no_input']
        data = self.data_list[index]
        data = json.loads(data)
        
        source = prompt_input.format_map(data) if 'input' in data else prompt_no_input.format_map(data)
        target = data['output']
        
        input_tokens = self.tokenizer.encode(source, add_special_tokens=False)
        output_tokens = self.tokenizer.encode(target, add_special_tokens=False) + [self.eos_token_id]
        
        input_ids = input_tokens + output_tokens
        target_mask = [0] * len(input_tokens) + [1] * len(output_tokens)

        assert len(input_ids) == len(target_mask)
        # truncate
        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask,
        }
        return inputs
        

class MixtralSFTDataset(Dataset):
    def __init__(self, file, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.max_seq_length = max_seq_length
        logging.info('Loading data: {}'.format(file))
        with open(file, 'r', encoding='utf8') as f:
            # data_list = json.load(f)
            data_list = f.readlines()
        
        logging.info("There are {} data in dataset".format(len(data_list)))
        self.data_list = data_list
        self.inst_begin_tokens = tokenizer.encode('[INST]', add_special_tokens=False)
        self.inst_end_tokens = tokenizer.encode('[/INST]', add_special_tokens=False)
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        '''
        <s>[INST]{instruction}[/INST]{answer}</s>
        '''
        prompt_input, prompt_no_input = PROMPT_DICT['prompt_input'], PROMPT_DICT['prompt_no_input']
        data = self.data_list[index]
        data = json.loads(data)
        
        
        # inputs
        input_ids = [self.bos_token_id]
        target_mask = [0]
        
        source = prompt_input.format_map(data) if 'input' in data else prompt_no_input.format_map(data)
        target = data['output']
        
        human_tokens = self.tokenizer.encode(source, add_special_tokens=False)
        assistant_tokens = self.tokenizer.encode(target, add_special_tokens=False)
        
        input_tokens = self.inst_begin_tokens + human_tokens + self.inst_end_tokens
        output_tokens = assistant_tokens + [self.eos_token_id]

        input_ids += input_tokens + output_tokens
        target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)

        assert len(input_ids) == len(target_mask)
        # truncate
        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }
        return inputs