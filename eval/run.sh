openai_key=sk-*
tasks=("ECARE:mc" "COPA:mc" "MCTACO:mc" "MCTACO:nli" "TRACIE:nli")
# tasks=("SocialIQA:mc" "ESTER:gen" "CQA:gen" "STC:mc" "MATRES:re" "ESL_D:re")
cuda=0

model_version=/data/WizardEvent/store/model/llama-2-7b-chat
model_name=llama2-7b
data_version=unimix_0205
version=0
K=0

for task in "${tasks[@]}"; do
    IFS=':' read -r task_name format <<< "$task"
    desc=${task_name}_${model_name}-${data_version}_${format}_K${K}_${version}
    echo ${desc}
    CUDA_VISIBLE_DEVICES=${cuda} python eval.py \
        --openai_key ${openai_key} \
        --cache_dir /data/WizardEvent/model/ \
        --model_version ${model_version} \
            --task_name ${task_name} \
            --generative_format ${format} \
        --desc ${desc} \
            --data_dir data/${task_name}/eve_2/${format}/ \
            --output_dir output/eve_v2/implicit/${desc}/ \
        --num_gpus 1 \
            --per_device_eval_batch_size 1 \
            --number_of_folds ${version} \
            --data_type ${K} 2>&1 | tee -a output/logs/output_${model_name}_${data_version}.txt
done



