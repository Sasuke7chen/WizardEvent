import os
import torch
from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging
from component.argument import LoraArguments
from component.dataset import LlamaSTFDataset, MixtralSFTDataset
from component.collator import STFDataCollator
from component.trainer import LoRATrainer
from component.loss import TargetLMLoss

def setup_everything():
    logging.info('Geting arguments...')
    parser = HfArgumentParser((LoraArguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    set_seed(training_args.seed)
    return args, training_args

def init_component(args, training_args):
    logging.info('Initializing components...')
    
    # load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.config.use_cache = False
        
    # load tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    
    # model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    print(f'memory footprint of model: {model.get_memory_footprint()/(1024*1024*1024)} GB')
    
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj",
            "v_proj"
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model.enable_input_require_grads()
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    # model.config.torch_dtype = torch.float32
    
    #Tell Trainer not to attempt DataParallel
    model.is_parallelizable = True
    model.model_parallel = True
    
    # define loss_func
    # loss_func = TargetLMLoss(ignore_index=-100)
    
    # load dataset
    if training_args.local_rank > 0: 
        torch.distributed.barrier()
        
    train_dataset = LlamaSTFDataset(args.train_file, tokenizer, args.max_seq_length)
    # train_dataset = MixtralSFTDataset(args.train_file, tokenizer, args.max_seq_length)
    
    if training_args.local_rank == 0: 
        torch.distributed.barrier()
        
    data_collator = STFDataCollator(tokenizer, args.max_seq_length)
    
    trainer = LoRATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # compute_loss=loss_func
    )
    return trainer

def main():
    args, training_args = setup_everything()
    trainer = init_component(args, training_args)
    
    logging.info("*** starting training ***")
    train_result = trainer.train()
    trainer.save_model(training_args.output_dir)
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
if __name__ == "__main__":
    main()
    