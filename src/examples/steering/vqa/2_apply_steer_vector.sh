#!/bin/bash

YOUR_XL_VLM_DIR=YOUR_XL_VLM_DIR
YOUR_XL_VLM_DIR=/home/khayatan/xl_vlms_cvpr/xl-vlms
YOUR_VQA_DIR=YOUR_VQA_DIR
YOUR_VQA_DIR=/data/mshukor/data/coco/
YOUR_ANSWER_TYPE_TO_ANSWER_FILE=YOUR_ANSWER_TYPE_TO_ANSWER_FILE
YOUR_ANSWER_TYPE_TO_ANSWER_FILE=/data/mshukor/data/coco/type_to_answer_dict.json

model_name_or_path=llava-hf/llava-1.5-7b-hf
model=llava

# model_name_or_path=HuggingFaceM4/idefics2-8b
# model=idefics2

# model_name_or_path=Qwen/Qwen2-VL-7B-Instruct
# model=qwen2vlinstruct

# model_name_or_path=allenai/Molmo-7B-D-0924
# model=molmo


dataset_name=vqav2
data_dir=YOUR_VQA_DIR
answer_type_to_answer=${YOUR_ANSWER_TYPE_TO_ANSWER_FILE}
dataset_size=5000
save_steer_dir=${YOUR_XL_VLM_DIR}/results/steering
save_dir=${YOUR_XL_VLM_DIR}/results
max_new_tokens=5

steering_method=shift_of_means
# steering_method=mean_of_shifts
steering_hook_name=shift_hidden_states_add
# steering_hook_name=shift_hidden_states_add_projected
# steering_hook_name=shift_hidden_states_add_only_generated
# steering_hook_name=shift_hidden_states_add_last_prompt_token

# start_prompt_token_idx_steering=577
# steering_hook_name=shift_hidden_states_add_start_idx
steering_alpha=1
analysis_name=steering_vector
start_prompt_token_idx_steering=0
token_of_interest=""
layers=(31)


category_of_interest=no
token_of_interests=("yes" "no")

# save_dataset_size=5000
# category_of_interest=number
# token_of_interests=("1" "3")

# save_dataset_size=5000
# category_of_interest=other
# token_of_interests=("white" "black")


for i in "${!layers[@]}"; do
    ## steering
    layer=${layers[$i]}
    modules_to_hook=language_model.model.layers.${layer}

    split=val2014
    annotation_file=v2_mscoco_${split}_annotations.json
    questions_file=v2_OpenEnded_mscoco_${split}_questions.json
    shift_vector_path=${save_steer_dir}/${steering_method}_${model}_${layer}_${token_of_interests[0]}_to_${token_of_interests[1]}.pth
    shift_vector_key=steering_vector
    hook_names=($steering_hook_name "vqav2_accuracy")
    token_of_interest=${token_of_interests[1]}
    python src/save_features.py \
    --model_name_or_path $model_name_or_path \
    --data_dir $data_dir \
    --annotation_file $annotation_file \
    --questions_file $questions_file \
    --answer_type_to_answer $answer_type_to_answer \
    --split $split \
    --dataset_size $dataset_size \
    --dataset_name $dataset_name \
    --hook_names "${hook_names[@]}" \
    --modules_to_hook $modules_to_hook \
    --token_of_interest_key answer_type \
    --token_of_interest $token_of_interest \
    --predictions_token_of_interest ${token_of_interests[1]} \
    --targets_token_of_interest ${token_of_interests[0]} \
    --generation_mode \
    --save_dir $save_dir \
    --save_filename ${steering_method}_${model}_${layer}_${token_of_interests[0]}_to_${token_of_interests[1]}_${shift_vector_key}_${steering_hook_name} \
    --local_files_only \
    --exact_match_modules_to_hook \
    --shift_vector_path $shift_vector_path \
    --shift_vector_key $shift_vector_key \
    --steering_alpha $steering_alpha \
    --category_of_interest $category_of_interest \
    --max_new_tokens $max_new_tokens \
    --start_prompt_token_idx_steering $start_prompt_token_idx_steering
done
