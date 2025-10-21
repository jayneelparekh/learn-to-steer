model_name_or_path=llava-hf/llava-1.5-7b-hf
model=llava


YOUR_FEAT_DIR=/data/khayatan/Hallucination/POPE/hallucination/features # TO BE REPLACED WITH YOUR SAVE DIR FOR POPE/features
YOUR_SAVE_DIR=/data/khayatan/Hallucination/POPE/hallucination/shift_vectors # TO BE REPLACED WITH YOUR SAVE DIR FOR POPE
YOUR_CACHE_DIR=/data/khayatan/cache/ # TO BE REPLACED WITH THE DIR OF YOUR MODEL


cache_dir=${YOUR_CACHE_DIR}
features_dir=${YOUR_FEAT_DIR}
save_dir=${YOUR_SAVE_DIR}

shift_type=average # We use the average representation of generated tokens to compute the shift vector

analysis_name=learnable_steering


for split in adversarial popular random; do

    for i in 14; do

        pos_features_name=save_hidden_states_for_l2s_llava_pope_test_features_pos_answers_${i}_${split}_-1.pth
        neg_features_name=save_hidden_states_for_l2s_llava_pope_test_features_neg_answers_${i}_${split}_-1.pth

        modules_to_hook="model.language_model.layers.${i};model.language_model.layers.${i}" # for previous transformer versions (4.47.1 for instance): language_model.model.layers.${i}

        save_filename=${split}_pope_test_-1


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





# for Qwen2vlinstruct


model_name_or_path=Qwen/Qwen2-VL-7B-Instruct
model=qwen2vlinstruct

YOUR_FEAT_DIR=/data/khayatan/Hallucination/POPE/hallucination/features # TO BE REPLACED WITH YOUR SAVE DIR FOR POPE/features
YOUR_SAVE_DIR=/data/khayatan/Hallucination/POPE/hallucination/shift_vectors # TO BE REPLACED WITH YOUR SAVE DIR FOR POPE
YOUR_CACHE_DIR=/data/khayatan/cache/ # TO BE REPLACED WITH THE DIR OF YOUR MODEL


cache_dir=${YOUR_CACHE_DIR}
features_dir=${YOUR_FEAT_DIR}
save_dir=${YOUR_SAVE_DIR}

shift_type=average
analysis_name=learnable_steering


for split in adversarial popular random; do

    for i in 17; do

        pos_features_name=save_hidden_states_for_l2s_qwen2vlinstruct_pope_test_features_pos_answers_${i}_${split}_-1.pth
        neg_features_name=save_hidden_states_for_l2s_qwen2vlinstruct_pope_test_features_neg_answers_${i}_${split}_-1.pth

        modules_to_hook="model.language_model.layers.${i};model.language_model.layers.${i}" # for previous transformer versions (4.47.1 for instance): model.layers.${i}
        save_filename=${split}_pope_test_-1


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
            --seed 0
    done
done






"""
Saving individual shift vectors in : 
/data/khayatan/Hallucination/POPE/hallucination/shift_vectors/llava_14_average_random_pope_test_-1.pth
/data/khayatan/Hallucination/POPE/hallucination/shift_vectors/llava_14_average_popular_pope_test_-1.pth
/data/khayatan/Hallucination/POPE/hallucination/shift_vectors/llava_14_average_adversarial_pope_test_-1.pth

/data/khayatan/Hallucination/POPE/hallucination/shift_vectors/qwen_17_average_random_pope_test_-1.pth
/data/khayatan/Hallucination/POPE/hallucination/shift_vectors/qwen_17_average_popular_pope_test_-1.pth
/data/khayatan/Hallucination/POPE/hallucination/shift_vectors/qwen_17_average_adversarial_pope_test_-1.pth
"""
