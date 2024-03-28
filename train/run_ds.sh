model_name=meta-llama/Llama-2-7b-hf
data_version=unimix
version=0205
deepspeed train.py \
    --model_name_or_path ${model_name} \
    --train_file data/semi_${version}/train_${data_version}.json \
    --output_dir store/ckpt/llama2-7b-${data_version}_${version} \
    --num_train_epochs 3 \
    --max_seq_length 512 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --logging_steps 100 \
    --weight_decay 0 \
    --max_grad_norm 0.3 \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --remove_unused_columns False \
    --deepspeed config/deepspeed_config2.json \
    --bf16 True \
    --seed 42 

