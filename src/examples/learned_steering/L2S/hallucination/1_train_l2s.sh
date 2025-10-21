model_name_or_path=llava-hf/llava-1.5-7b-hf
model=llava


YOUR_FEAT_DIR=/data/khayatan/Hallucination/POPE/hallucination/features # TO BE REPLACED WITH YOUR SAVE DIR FOR POPE/features
YOUR_SAVE_DIR=/data/khayatan/Hallucination/POPE/hallucination/shift_vectors # TO BE REPLACED WITH YOUR SAVE DIR FOR POPE


features_dir=${YOUR_FEAT_DIR}
save_dir=${YOUR_SAVE_DIR}


dataset_name=pope_train
dataset_size=-1


shift_type=average
analysis_name=learnable_steering


for split in all; do

    for i in 14; do

        pos_features_name=save_hidden_states_for_l2s_${model}_pope_train_${split}_features_pos_answers_14_${dataset_size}.pth
        neg_features_name=save_hidden_states_for_l2s_${model}_pope_train_${split}_features_neg_answers_14_${dataset_size}.pth

        modules_to_hook="model.language_model.layers.${i};model.language_model.layers.${i}" # for previous transformer versions (4.47.1 for instance): language_model.model.layers.${i}

        save_filename=${split}_pope_train_-1


        python src/analyse_features.py \
            --model_name_or_path $model_name_or_path \
            --save_dir $save_dir \
            --analysis_name $analysis_name \
            --modules_to_hook $modules_to_hook \
            --save_filename ${save_filename} \
            --local_files_only \
            --shift_type $shift_type \
            --features_path ${features_dir}/${pos_features_name} ${features_dir}/${neg_features_name} \
            --seed 0
    done
done













model_name_or_path=Qwen/Qwen2-VL-7B-Instruct
model=qwen2vlinstruct

YOUR_FEAT_DIR=/data/khayatan/Hallucination/POPE/hallucination/features # TO BE REPLACED WITH YOUR SAVE DIR FOR POPE/features
YOUR_SAVE_DIR=/data/khayatan/Hallucination/POPE/hallucination/shift_vectors # TO BE REPLACED WITH YOUR SAVE DIR FOR POPE
YOUR_CACHE_DIR=/data/khayatan/cache/ # TO BE REPLACED WITH THE DIR OF YOUR MODEL


cache_dir=${YOUR_CACHE_DIR}
features_dir=${YOUR_FEAT_DIR}
save_dir=${YOUR_SAVE_DIR}

dataset_name=pope_train
dataset_size=-1


shift_type=average
analysis_name=learnable_steering


for split in all; do

    for i in 17; do
        pos_features_name=save_hidden_states_for_l2s_${model}_pope_train_${split}_features_pos_answers_${i}_${dataset_size}.pth
        neg_features_name=save_hidden_states_for_l2s_${model}_pope_train_${split}_features_neg_answers_${i}_${dataset_size}.pth

        modules_to_hook="model.language_model.layers.${i};model.language_model.layers.${i}" # for previous transformer versions (4.47.1 for instance): model.layers.${i}


        save_filename=${split}_pope_train_-1


        python src/analyse_features.py \
            --model_name_or_path $model_name_or_path \
            --cache_dir $cache_dir \
            --save_dir $save_dir \
            --analysis_name $analysis_name \
            --modules_to_hook $modules_to_hook \
            --save_filename ${save_filename} \
            --local_files_only \
            --shift_type $shift_type \
            --features_path ${features_dir}/${pos_features_name} ${features_dir}/${neg_features_name} \
            --seed 0 \
            --hidden_size 400
    done
done


