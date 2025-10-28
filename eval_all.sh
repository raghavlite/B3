#!/bin/bash

datasets=(Wiki-SS-NQ VisDial CIRR VisualNews_t2i VisualNews_i2t MSCOCO_t2i MSCOCO_i2t NIGHTS WebQA OVEN FashionIQ EDIS OK-VQA A-OKVQA DocVQA InfographicsVQA ChartQA Visual7W ScienceQA GQA TextVQA VizWiz ImageNet-1K HatefulMemes SUN397 N24News VOC2007 Place365 ImageNet-A ImageNet-R ObjectNet Country211 MSCOCO RefCOCO RefCOCO-Matching Visual7W-Pointing)
# datasets=(VisDial VisualNews_t2i VisualNews_i2t MSCOCO_t2i MSCOCO_i2t Visual7W MSCOCO)
# extras=(CIRR NIGHTS WebQA OK-VQA A-OKVQA DocVQA InfographicsVQA ChartQA ImageNet_1K HatefulMemes SUN397 N24News VOC2007)

convert_to_decimal() {
    local input_number=$1
    echo $((10#$input_number))
}
formatted_task_id=$(printf "%02d" ${SLURM_ARRAY_TASK_ID})


job_id=$SLURM_JOB_ID

# Calculate the job ID modulo 100
mod_job_id=$((job_id % 100))


index=$(convert_to_decimal $formatted_task_id)
# index=5
# port=$((25500+index+mod_job_id))
model=$1
output_path=$2

task=${datasets[index]}

# Define the array
echo "Dataset at index $index: $task: $model, ${output_path}"
# accelerate launch --main_process_port ${port} eval_mteb_gritlm2.py --model_name_or_path ${model} --instruction_set ${instruction_set} --instruction_format ${instruction_format} --task_names ${task} --attn bbcc  --batch_size 32




# ./run_inf.sh python eval_mmeb.py --model_name ${model} --encode_output_path ${output_path} --pooling eos --normalize True --lora --lora_r 8 --bf16 --dataset_name TIGER-Lab/MMEB-eval --subset_name ${task} --dataset_split test --per_device_eval_batch_size 4 --image_dir ../VLM2Vec/MMEB-eval/eval_images/
./run_inf.sh python eval_mmeb.py --model_name ${model} --encode_output_path ${output_path} --pooling eos --normalize True --lora --lora_r 8 --bf16 --dataset_name TIGER-Lab/MMEB-eval --subset_name ${task} --dataset_split test --per_device_eval_batch_size 4 --image_dir ../VLM2Vec/MMEB-eval/eval_images/ --tgt_prefix_mod
# ./run_inf.sh python eval_mmeb.py --model_name ${model} --encode_output_path ${output_path} --pooling eos --normalize True --bf16 --dataset_name TIGER-Lab/MMEB-eval --subset_name ${task} --dataset_split test --per_device_eval_batch_size 4 --image_dir ../VLM2Vec/MMEB-eval/eval_images/ --tgt_prefix_mod
