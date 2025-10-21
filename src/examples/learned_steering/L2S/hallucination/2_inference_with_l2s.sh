model_name_or_path=llava-hf/llava-1.5-7b-hf
model=llava


YOUR_DATA_DIR=/data/khayatan/datasets/POPE/test
YOUR_SAVE_DIR=/data/khayatan/Hallucination/POPE/hallucination
STEER_MODEL_NAME=llava_14_average_all_pope_train_-1.pt



data_dir=${YOUR_DATA_DIR}
save_dir=${YOUR_SAVE_DIR}
steer_model_base=$(basename "$STEER_MODEL_NAME" .pt)



dataset_name=pope_test
dataset_size=-1
max_new_tokens=128
steering_alpha=1
hook_names=("shift_hidden_states_learned_steer" "hallucination_metrics") # should add the evaluation right here



for split in adversarial popular random; do


    for i in 14; do
        shift_vector_path=${STEER_MODEL_NAME}
        save_filename="${model}_${dataset_name}_steer_${i}_yes_no_${split}_${steering_alpha}_${steer_model_base}"
        modules_to_hook="model.language_model.layers.${i}" # for previous transformer versions (4.47.1 for instance): language_model.model.layers.${i}



        python src/save_features.py \
            --model_name_or_path $model_name_or_path \
            --save_dir $save_dir \
            --data_dir $data_dir \
            --split $split \
            --dataset_size $dataset_size \
            --dataset_name $dataset_name \
            --hook_names "${hook_names[@]}" \
            --modules_to_hook $modules_to_hook \
            --generation_mode \
            --save_filename $save_filename \
            --save_predictions \
            --local_files_only \
            --exact_match_modules_to_hook \
            --shift_vector_path $shift_vector_path \
            --steering_alpha $steering_alpha \
            --individual_shift \
            --max_new_tokens $max_new_tokens \
            --seed 0
    done
done







model_name_or_path=Qwen/Qwen2-VL-7B-Instruct
model=qwen2vlinstruct

YOUR_DATA_DIR=/data/khayatan/datasets/POPE/test
YOUR_SAVE_DIR=/data/khayatan/Hallucination/POPE/hallucination
STEER_MODEL_NAME=qwen2vlinstruct_17_average_all_pope_train_-1.pt
# STEER_MODEL_NAME=/home/khayatan/learnable_steering/xl-vlms/0.0001_1_5e-05_last_input_average_400_17.pt
steer_model_base=$(basename "$STEER_MODEL_NAME" .pt)
YOUR_CACHE_DIR=/data/khayatan/cache/


cache_dir=${YOUR_CACHE_DIR}
data_dir=${YOUR_DATA_DIR}
save_dir=${YOUR_SAVE_DIR}


dataset_name=pope_test
dataset_size=-1
max_new_tokens=128
steering_alpha=1
hook_names=("shift_hidden_states_learned_steer" "hallucination_metrics") # should add the evaluation right here



for split in adversarial popular random; do

    for i in 17; do
        shift_vector_path=${STEER_MODEL_NAME}
        save_filename="${model}_${dataset_name}_steer_${i}_yes_no_${split}_${steering_alpha}_${steer_model_base}"
        modules_to_hook="model.language_model.layers.${i}" # for previous transformer versions (4.47.1 for instance): model.layers.${i}


        python src/save_features.py \
            --model_name_or_path $model_name_or_path \
            --cache_dir $cache_dir \
            --save_dir $save_dir \
            --data_dir $data_dir \
            --split $split \
            --dataset_size $dataset_size \
            --dataset_name $dataset_name \
            --hook_names "${hook_names[@]}" \
            --modules_to_hook $modules_to_hook \
            --generation_mode \
            --save_filename $save_filename \
            --save_predictions \
            --local_files_only \
            --exact_match_modules_to_hook \
            --shift_vector_path $shift_vector_path \
            --steering_alpha $steering_alpha \
            --individual_shift \
            --max_new_tokens $max_new_tokens \
            --seed 0 \
            --hidden_size 400
    done
done



