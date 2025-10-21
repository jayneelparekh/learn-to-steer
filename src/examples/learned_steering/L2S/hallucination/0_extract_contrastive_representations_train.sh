model_name_or_path=llava-hf/llava-1.5-7b-hf
model=llava


YOUR_DATA_DIR=/data/khayatan/datasets/POPE/train
YOUR_SAVE_DIR=/data/khayatan/Hallucination/POPE/hallucination


data_dir=${YOUR_DATA_DIR}
save_dir=${YOUR_SAVE_DIR}


dataset_name=pope_train
dataset_size=-1

max_new_tokens=100


hook_names=("save_hidden_states_for_l2s")
modules_to_hook=""

# individual splits of the pope dataset adversarial popular random
for split in all; do

    for i in 14; do

        modules_to_hook="model.language_model.layers.${i}" # for previous transformer versions (4.47.1 for instance): language_model.model.layers.${i}
        save_filename="${model}_${dataset_name}_${split}_features_pos_answers_${i}_${dataset_size}"


        python src/save_features.py \
            --model_name_or_path $model_name_or_path \
            --data_dir $data_dir \
            --dataset_name $dataset_name \
            --split $split \
            --annotation_file annotations.json \
            --dataset_size $dataset_size \
            --save_dir $save_dir \
            --max_new_tokens $max_new_tokens \
            --hook_names $hook_names \
            --modules_to_hook $modules_to_hook \
            --generation_mode \
            --save_filename ${save_filename} \
            --local_files_only \
            --force_answer \
            --forced_answer_true \
            --exact_match_modules_to_hook \
            --end_special_tokens "</s>" \
            --seed 0
    done
done



for split in all; do

    for i in 14; do

        modules_to_hook="model.language_model.layers.${i}" # for previous transformer versions (4.47.1 for instance): language_model.model.layers.${i}
        save_filename="${model}_${dataset_name}_${split}_features_neg_answers_${i}_${dataset_size}"


        python src/save_features.py \
            --model_name_or_path $model_name_or_path \
            --data_dir $data_dir \
            --dataset_name $dataset_name \
            --dataset_size $dataset_size \
            --split $split \
            --save_dir $save_dir \
            --max_new_tokens $max_new_tokens \
            --hook_names $hook_names \
            --modules_to_hook $modules_to_hook \
            --generation_mode \
            --save_filename ${save_filename} \
            --local_files_only \
            --force_answer \
            --exact_match_modules_to_hook \
            --end_special_tokens "</s>" \
            --seed 0
    done
done












model_name_or_path=Qwen/Qwen2-VL-7B-Instruct
model=qwen2vlinstruct

YOUR_DATA_DIR=/data/khayatan/datasets/POPE/train
YOUR_SAVE_DIR=/data/khayatan/Hallucination/POPE/hallucination
YOUR_CACHE_DIR=/data/khayatan/cache/


cache_dir=${YOUR_CACHE_DIR}
data_dir=${YOUR_DATA_DIR}
save_dir=${YOUR_SAVE_DIR}

dataset_name=pope_train
dataset_size=-1

max_new_tokens=100


hook_names=("save_hidden_states_for_l2s")
modules_to_hook=""

# individual splits of the pope dataset adversarial popular random
for split in all; do

    for i in 17; do

        modules_to_hook="model.language_model.layers.${i}" # for previous transformer versions (4.47.1 for instance): model.layers.${i}
        save_filename="${model}_${dataset_name}_${split}_features_pos_answers_${i}_${dataset_size}"


        python src/save_features.py \
            --model_name_or_path $model_name_or_path \
            --cache_dir $cache_dir \
            --data_dir $data_dir \
            --dataset_name $dataset_name \
            --split $split \
            --annotation_file annotations.json \
            --dataset_size $dataset_size \
            --save_dir $save_dir \
            --max_new_tokens $max_new_tokens \
            --hook_names $hook_names \
            --modules_to_hook $modules_to_hook \
            --generation_mode \
            --save_filename ${save_filename} \
            --local_files_only \
            --force_answer \
            --forced_answer_true \
            --exact_match_modules_to_hook \
            --end_special_tokens "</s>" \
            --seed 0
    done
done



for split in all; do

    for i in 17; do

        modules_to_hook="model.language_model.layers.${i}" # for previous transformer versions (4.47.1 for instance): model.layers.${i}
        save_filename="${model}_${dataset_name}_${split}_features_neg_answers_${i}_${dataset_size}"


        python src/save_features.py \
            --model_name_or_path $model_name_or_path \
            --cache_dir $cache_dir \
            --data_dir $data_dir \
            --dataset_name $dataset_name \
            --dataset_size $dataset_size \
            --split $split \
            --save_dir $save_dir \
            --max_new_tokens $max_new_tokens \
            --hook_names $hook_names \
            --modules_to_hook $modules_to_hook \
            --generation_mode \
            --save_filename ${save_filename} \
            --local_files_only \
            --force_answer \
            --exact_match_modules_to_hook \
            --end_special_tokens "</s>" \
            --seed 0
    done
done



