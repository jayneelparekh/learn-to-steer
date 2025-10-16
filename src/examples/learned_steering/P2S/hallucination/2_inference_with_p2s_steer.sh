model_name_or_path=llava-hf/llava-1.5-7b-hf
model=llava

YOUR_DATA_DIR=/data/khayatan/datasets/POPE/test
YOUR_SAVE_DIR=/data/khayatan/Hallucination/POPE/hallucination
YOUR_CACHE_DIR=/data/khayatan/cache/


cache_dir=${YOUR_CACHE_DIR}
data_dir=${YOUR_DATA_DIR}
save_dir=${YOUR_SAVE_DIR}



# the script for adding and evaluating the mean steering vector is in : learn-to-steer/src/examples/learned_steering/L2S/hallucination/3_inference_with_mean_steer.sh

dataset_name=pope_test
dataset_size=-1
max_new_tokens=128
steering_alpha=1
shift_type=average

hook_names=("shift_hidden_states_add" "hallucination_metrics")
shift_vector_key=steering_vector


for subset in adversarial popular random; do

    for steering_alpha in 1; do

        for i in 14; do
            shift_vector_path=${save_dir}/shift_vectors/${model}_${i}_${shift_type}_${subset}_${dataset_name}_${dataset_size}.pth
            save_filename="${model}_${dataset_name}_steer_${i}_yes_no_${subset}_${steering_alpha}_p2s"
            modules_to_hook="model.language_model.layers.${i}" # for previous transformer versions (4.47.1 for instance): language_model.model.layers.${i}


            python src/save_features.py \
                --model_name_or_path $model_name_or_path \
                --save_dir $save_dir \
                --data_dir $data_dir \
                --split $subset \
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
                --shift_vector_key $shift_vector_key \
                --steering_alpha $steering_alpha \
                --individual_shift \
                --max_new_tokens $max_new_tokens \
                --seed 0
        done
    done
done






# for Qwen2vlinstruct
model_name_or_path=Qwen/Qwen2-VL-7B-Instruct
model=qwen2vlinstruct



# the script for adding and evaluating the mean steering vector is in : learn-to-steer/src/examples/learned_steering/L2S/hallucination/3_inference_with_mean_steer.sh

dataset_name=pope_test
dataset_size=-1
max_new_tokens=128
steering_alpha=1
hook_names=("shift_hidden_states_add" "hallucination_metrics")
shift_vector_key=steering_vector


for subset in adversarial popular random; do

    for steering_alpha in 1; do

        for i in 17; do
            shift_vector_path=/data/khayatan/Hallucination/POPE/hallucination/shift_vectors/${model}_${i}_average_${subset}_${dataset_name}_${dataset_size}.pth
            save_filename="${model}_${dataset_name}_steer_${i}_yes_no_${subset}_${steering_alpha}_p2s"
            modules_to_hook="model.language_model.layers.${i}" # for previous transformer versions (4.47.1 for instance): model.layers.${i}


            python src/save_features.py \
                --model_name_or_path $model_name_or_path \
                --cache_dir $cache_dir \
                --save_dir $save_dir \
                --data_dir $data_dir \
                --split $subset \
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
                --shift_vector_key $shift_vector_key \
                --steering_alpha $steering_alpha \
                --individual_shift \
                --max_new_tokens $max_new_tokens \
                --seed 0
        done
    done
done




