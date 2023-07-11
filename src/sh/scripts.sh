### MULTITASK Full Dataset.

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_multi_gen.py --task summarize \
--log_steps 100 --num_saves 5 --max_source_length -1 --max_target_length -1 \
--data_num -1 --warmup_steps 1000 --learning_rate 5e-5 --patience -1  \
--save_last_checkpoints --always_save_model --num_saves=10 \
--num_train_epochs 15 --train_batch_size 16 --eval_batch_size 16 \
--name multitask_all --project_name summ_exp3



#####  Continual Summarization
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --log_steps=10 --data_num=-1 --full_matrix_eval --warmup_steps=1000 --save_last_checkpoints --always_save_model --project_name=summ_exp3 --name=all --stream=summarize



##### #####  Continual Tasks  ##### #####
### No task and lang IDS
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --log_steps=10 --data_num=-1 --warmup_steps=1000 --save_last_checkpoints --always_save_model --project_name=task_exp2 --name=CSTR --stream=concode_none,translate_java-cs,summarize_ruby,refine_small
### With task and lang IDS
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --add_task_prefix --add_lang_ids --log_steps=10 --data_num=-1 --warmup_steps=1000 --save_last_checkpoints --always_save_model --project_name=task_exp2 --name=CSTR_ids --stream=concode_none,translate_java-cs,summarize_ruby,refine_small
### With task ids and lang ids and replay
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --replay=ring --buffer_size=5000 --add_task_prefix --add_lang_ids --log_steps=10 --data_num=-1 --warmup_steps=1000 --save_last_checkpoints --always_save_model --project_name=task_exp2 --name=CSTR_ids_replay5k --stream=concode_none,translate_java-cs,summarize_ruby,refine_small


#### Task specific Prompts Tuning####
### Tune only prompt - No task tag added. ###
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --num_prompts_per_task=20 --train_only_prompts --log_steps=10 --data_num=-1 --warmup_steps=1000 --save_last_checkpoints --always_save_model --project_name=task_exp2 --name=CSTR_onlyprompt --stream=concode_none,translate_java-cs,summarize_ruby,refine_small
### Tune prompt and T5 ###
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --train_batch_size=8 --eval_batch_size=32 --num_prompts_per_task=100 --log_steps=10 --data_num=-1 --warmup_steps=1000 --save_last_checkpoints --always_save_model --project_name=task_exp2 --name=CSTR_tuned_prompt_all --stream=concode_none,translate_java-cs,summarize_ruby,refine_small
### Tune Prompt and T5 + REPLAY ###
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --train_batch_size=8 --eval_batch_size=32 --num_prompts_per_task=100 --replay=ring --buffer_size=1000 --log_steps=10 --data_num=-1 --warmup_steps=1000 --save_last_checkpoints --always_save_model --project_name=task_exp2 --name=CSTR_tuned_prompt_replay --stream=concode_none,translate_java-cs,summarize_ruby,refine_small
### prompt Pool ###
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --train_batch_size=8 --eval_batch_size=32 --prompt_pool --prompt_projection --projection_out_dim=64 --pool_lambda=0.5 --pool_size=60 --num_pool_prompt_tokens=5 --num_prompts_per_task=20 --train_only_prompts --log_steps=10 --data_num=-1 --warmup_steps=50 --save_last_checkpoints --always_save_model --project_name=task_exp2 --name=CSTR_pool_proj --stream=concode_none,translate_java-cs,summarize_ruby,refine_small
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --train_batch_size=8 --eval_batch_size=32 --prompt_pool --prompt_projection --projection_out_dim=64 --pool_freq_norm --pool_lambda=0.5 --pool_size=60 --num_pool_prompt_tokens=5 --num_prompts_per_task=20 --train_only_prompts --log_steps=10 --data_num=-1 --warmup_steps=50 --save_last_checkpoints --always_save_model --project_name=task_exp2 --name=CSTR_pool_proj_freq_norm --stream=concode_none,translate_java-cs,summarize_ruby,refine_small

### MULTITASK LEARNING ####
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_multi_gen.py --task=all_tasks --log_steps=100 --num_saves=5 --max_source_length=-1 --max_target_length=-1 --data_num=-1 --warmup_steps=1000 --learning_rate=5e-5 --patience=-1 --save_last_checkpoints --always_save_model --train_batch_size=16 --eval_batch_size=100 --name=multitask_tasks --project_name=task_exp2









##### POOL + Replay ####
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --replay=ring --buffer_size=100 --prompt_pool --separate_projection --train_only_prompts --prompt_projection --projection_hid_dim=512 --projection_out_dim=64 --pool_lambda=0.5 --pool_size=60 --num_pool_prompt_tokens=5 --num_prompts_per_task=20  --train_batch_size=8 --eval_batch_size=32 --log_steps=10 --data_num=-1 --warmup_steps=10 --save_last_checkpoints --always_save_model --project_name=task_exp2 --name=CSTR_tuned_prompt_replay --stream=concode_none,translate_java-cs,summarize_ruby,refine_small
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --prompt_pool --num_prompts_per_task=20 --train_only_prompts --bleu_samples=100 --num_train_epochs=30 --patience=100 --warmup_steps=0 --train_batch_size=8 --eval_batch_size=32 --log_steps=10 --data_num=1000 --save_last_checkpoints --always_save_model --project_name=debug_pool_replay --name=pool --stream=concode_none,translate_java-cs,summarize_ruby,refine_small
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --prompt_projection --projection_hid_dim=512 --projection_out_dim=64 --prompt_pool --num_prompts_per_task=20 --train_only_prompts --bleu_samples=100 --num_train_epochs=30 --patience=100 --warmup_steps=0 --train_batch_size=8 --eval_batch_size=32 --log_steps=10 --data_num=1000 --save_last_checkpoints --always_save_model --project_name=debug_pool_replay --name=pool_shared_proj --stream=concode_none,translate_java-cs,summarize_ruby,refine_small
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --prompt_projection --separate_projection --projection_hid_dim=512 --projection_out_dim=64 --prompt_pool --num_prompts_per_task=20 --train_only_prompts --bleu_samples=100 --num_train_epochs=30 --patience=100 --warmup_steps=0 --train_batch_size=8 --eval_batch_size=32 --log_steps=10 --data_num=1000 --save_last_checkpoints --always_save_model --project_name=debug_pool_replay --name=pool_sep_proj --stream=concode_none,translate_java-cs,summarize_ruby,refine_small

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --num_prompts_per_task=100 --train_only_prompts --bleu_samples=100 --num_train_epochs=30 --patience=100 --warmup_steps=0 --train_batch_size=8 --eval_batch_size=32 --log_steps=10 --data_num=1000 --save_last_checkpoints --always_save_model --project_name=debug_pool_replay --name=tspt --stream=concode_none,translate_java-cs,summarize_ruby,refine_small
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --replay=ring --buffer_size=100 --num_prompts_per_task=100 --bleu_samples=100 --num_train_epochs=30 --patience=100 --warmup_steps=0 --train_batch_size=8 --eval_batch_size=32 --log_steps=10 --data_num=1000 --save_last_checkpoints --always_save_model --project_name=debug_pool_replay --name=tspt_replay --stream=concode_none,translate_java-cs,summarize_ruby,refine_small

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --replay=ring --buffer_size=100 --prompt_pool --num_prompts_per_task=20 --train_only_prompts --bleu_samples=100 --num_train_epochs=30 --patience=100 --warmup_steps=0 --train_batch_size=8 --eval_batch_size=32 --log_steps=10 --data_num=1000 --save_last_checkpoints --always_save_model --project_name=debug_pool_replay --name=pool_replay --stream=concode_none,translate_java-cs,summarize_ruby,refine_small
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --replay=ring --buffer_size=100 --prompt_projection --projection_hid_dim=512 --projection_out_dim=64 --prompt_pool --num_prompts_per_task=20 --train_only_prompts --bleu_samples=100 --num_train_epochs=30 --patience=100 --warmup_steps=0 --train_batch_size=8 --eval_batch_size=32 --log_steps=10 --data_num=1000 --save_last_checkpoints --always_save_model --project_name=debug_pool_replay --name=pool_shared_proj_replay --stream=concode_none,translate_java-cs,summarize_ruby,refine_small
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --replay=ring --buffer_size=100 --prompt_projection --separate_projection --projection_hid_dim=512 --projection_out_dim=64 --prompt_pool --num_prompts_per_task=20 --train_only_prompts --bleu_samples=100 --num_train_epochs=30 --patience=100 --warmup_steps=0 --train_batch_size=8 --eval_batch_size=32 --log_steps=10 --data_num=1000 --save_last_checkpoints --always_save_model --project_name=debug_pool_replay --name=pool_sep_proj_replay --stream=concode_none,translate_java-cs,summarize_ruby,refine_small

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --replay=ring --buffer_size=100 --prompt_pool --num_prompts_per_task=20 --bleu_samples=100 --num_train_epochs=30 --patience=100 --warmup_steps=0 --train_batch_size=8 --eval_batch_size=32 --log_steps=10 --data_num=1000 --save_last_checkpoints --always_save_model --project_name=debug_pool_replay --name=allparams_pool_replay --stream=concode_none,translate_java-cs,summarize_ruby,refine_small
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --replay=ring --buffer_size=100 --prompt_projection --projection_hid_dim=512 --projection_out_dim=64 --prompt_pool --num_prompts_per_task=20 --bleu_samples=100 --num_train_epochs=30 --patience=100 --warmup_steps=0 --train_batch_size=8 --eval_batch_size=32 --log_steps=10 --data_num=1000 --save_last_checkpoints --always_save_model --project_name=debug_pool_replay --name=allparams_pool_shared_proj_replay --stream=concode_none,translate_java-cs,summarize_ruby,refine_small
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --replay=ring --buffer_size=100 --prompt_projection --separate_projection --projection_hid_dim=512 --projection_out_dim=64 --prompt_pool --num_prompts_per_task=20 --bleu_samples=100 --num_train_epochs=30 --patience=100 --warmup_steps=0 --train_batch_size=8 --eval_batch_size=32 --log_steps=10 --data_num=1000 --save_last_checkpoints --always_save_model --project_name=debug_pool_replay --name=allparams_pool_sep_proj_replay --stream=concode_none,translate_java-cs,summarize_ruby,refine_small

### POOL TUNE LAMBDA
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --prompt_projection --separate_projection --pool_lambda=0.1 --projection_hid_dim=512 --projection_out_dim=64 --prompt_pool --num_prompts_per_task=20 --train_only_prompts --bleu_samples=100 --num_train_epochs=30 --patience=100 --warmup_steps=0 --train_batch_size=8 --eval_batch_size=32 --log_steps=10 --data_num=1000 --save_last_checkpoints --always_save_model --project_name=tune_pool_lambda --name=lambda_0.1 --stream=concode_none,translate_java-cs,summarize_ruby,refine_small
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --prompt_projection --separate_projection --pool_lambda=0.3 --projection_hid_dim=512 --projection_out_dim=64 --prompt_pool --num_prompts_per_task=20 --train_only_prompts --bleu_samples=100 --num_train_epochs=30 --patience=100 --warmup_steps=0 --train_batch_size=8 --eval_batch_size=32 --log_steps=10 --data_num=1000 --save_last_checkpoints --always_save_model --project_name=tune_pool_lambda --name=lambda_0.3 --stream=concode_none,translate_java-cs,summarize_ruby,refine_small
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --prompt_projection --separate_projection --pool_lambda=0.5 --projection_hid_dim=512 --projection_out_dim=64 --prompt_pool --num_prompts_per_task=20 --train_only_prompts --bleu_samples=100 --num_train_epochs=30 --patience=100 --warmup_steps=0 --train_batch_size=8 --eval_batch_size=32 --log_steps=10 --data_num=1000 --save_last_checkpoints --always_save_model --project_name=tune_pool_lambda --name=lambda_0.5 --stream=concode_none,translate_java-cs,summarize_ruby,refine_small
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --prompt_projection --separate_projection --pool_lambda=0.7 --projection_hid_dim=512 --projection_out_dim=64 --prompt_pool --num_prompts_per_task=20 --train_only_prompts --bleu_samples=100 --num_train_epochs=30 --patience=100 --warmup_steps=0 --train_batch_size=8 --eval_batch_size=32 --log_steps=10 --data_num=1000 --save_last_checkpoints --always_save_model --project_name=tune_pool_lambda --name=lambda_0.7 --stream=concode_none,translate_java-cs,summarize_ruby,refine_small
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --prompt_projection --separate_projection --pool_lambda=1 --projection_hid_dim=512 --projection_out_dim=64 --prompt_pool --num_prompts_per_task=20 --train_only_prompts --bleu_samples=100 --num_train_epochs=30 --patience=100 --warmup_steps=0 --train_batch_size=8 --eval_batch_size=32 --log_steps=10 --data_num=1000 --save_last_checkpoints --always_save_model --project_name=tune_pool_lambda --name=lambda_1 --stream=concode_none,translate_java-cs,summarize_ruby,refine_small
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --prompt_projection --separate_projection --pool_lambda=2 --projection_hid_dim=512 --projection_out_dim=64 --prompt_pool --num_prompts_per_task=20 --train_only_prompts --bleu_samples=100 --num_train_epochs=30 --patience=100 --warmup_steps=0 --train_batch_size=8 --eval_batch_size=32 --log_steps=10 --data_num=1000 --save_last_checkpoints --always_save_model --project_name=tune_pool_lambda --name=lambda_2 --stream=concode_none,translate_java-cs,summarize_ruby,refine_small


source sh/runs4.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --train_batch_size=8 --eval_batch_size=32 --prompt_pool --prompt_projection --projection_hid_dim=512 --projection_out_dim=64 --pool_lambda=0.5 --pool_size=60 --num_pool_prompt_tokens=5 --num_prompts_per_task=20 --train_only_prompts --log_steps=10 --data_num=1000 --bleu_samples=100 --num_train_epochs=30 --patience=100 --warmup_steps=0 --save_last_checkpoints --always_save_model --project_name=debug_pool_IOQ --name=1000_IOQ_proj_contract --stream=concode_none,translate_java-cs,summarize_ruby,refine_small






### Pooling Experiments
## MTL 1000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_multi_gen.py --task=all_tasks --log_steps=100 --num_saves=5 --max_source_length=-1 --max_target_length=-1 --data_num=1000 --warmup_steps=1000 --learning_rate=5e-5 --patience=-1 --save_last_checkpoints --always_save_model --train_batch_size=16 --eval_batch_size=100 --name=1000_multitask_tasks --project_name=debug_pool1

### test input output to query
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --train_batch_size=8 --eval_batch_size=32 --prompt_pool --prompt_projection --projection_hid_dim=512 --projection_out_dim=64 --pool_lambda=0.5 --pool_size=60 --num_pool_prompt_tokens=5 --num_prompts_per_task=20 --train_only_prompts --log_steps=10 --data_num=1000 --bleu_samples=100 --num_train_epochs=30 --patience=100 --warmup_steps=0 --save_last_checkpoints --always_save_model --project_name=debug_pool_IOQ --name=1000_IOQ_proj_contract --stream=concode_none,translate_java-cs,summarize_ruby,refine_small
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --train_batch_size=8 --eval_batch_size=32 --prompt_pool --separate_projection --prompt_projection --projection_hid_dim=512 --projection_out_dim=64 --pool_lambda=0.5 --pool_size=60 --num_pool_prompt_tokens=5 --num_prompts_per_task=20 --train_only_prompts --log_steps=10 --data_num=1000 --bleu_samples=100 --num_train_epochs=30 --patience=100 --warmup_steps=0 --save_last_checkpoints --always_save_model --project_name=debug_pool_IOQ --name=1000_IOQ_sep_proj_contract --stream=concode_none,translate_java-cs,summarize_ruby,refine_small
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --train_batch_size=8 --eval_batch_size=32 --prompt_pool --pool_lambda=0.5 --pool_size=60 --num_pool_prompt_tokens=5 --num_prompts_per_task=20 --train_only_prompts --log_steps=10 --data_num=1000 --bleu_samples=100 --num_train_epochs=30 --patience=100 --warmup_steps=0 --save_last_checkpoints --always_save_model --project_name=debug_pool_IOQ --name=1000_IOQ --stream=concode_none,translate_java-cs,summarize_ruby,refine_small


#p31, p32, p33, p34
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --train_batch_size=8 --eval_batch_size=32 --prompt_pool --prompt_projection --projection_hid_dim=512 --projection_out_dim=64 --pool_lambda=0.5 --pool_size=60 --num_pool_prompt_tokens=5 --num_prompts_per_task=20 --train_only_prompts --log_steps=10 --data_num=1000 --bleu_samples=100 --num_train_epochs=30 --patience=100 --warmup_steps=0 --save_last_checkpoints --always_save_model --project_name=debug_pool1 --name=1000_proj_contract --stream=concode_none,translate_java-cs,summarize_ruby,refine_small
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python cont_gen.py --train_batch_size=8 --eval_batch_size=32 --prompt_pool --prompt_projection --projection_hid_dim=2048 --projection_out_dim=2048 --pool_lambda=0.5 --pool_size=60 --num_pool_prompt_tokens=5 --num_prompts_per_task=20 --train_only_prompts --log_steps=10 --data_num=1000 --bleu_samples=100 --num_train_epochs=30 --patience=100 --warmup_steps=0 --save_last_checkpoints --always_save_model --project_name=debug_pool1 --name=1000_proj_expansion --stream=concode_none,translate_java-cs,summarize_ruby,refine_small
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --train_batch_size=8 --eval_batch_size=32 --prompt_pool --pool_freq_norm --prompt_projection --projection_hid_dim=512 --projection_out_dim=64 --pool_lambda=0.5 --pool_size=60 --num_pool_prompt_tokens=5 --num_prompts_per_task=20 --train_only_prompts --log_steps=10 --data_num=1000 --bleu_samples=100 --num_train_epochs=30 --patience=100 --warmup_steps=0 --save_last_checkpoints --always_save_model --project_name=debug_pool1 --name=1000_fnorm_proj_contract --stream=concode_none,translate_java-cs,summarize_ruby,refine_small
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --train_batch_size=8 --eval_batch_size=32 --prompt_pool --pool_freq_norm --prompt_projection --projection_hid_dim=2048 --projection_out_dim=2048 --pool_lambda=0.5 --pool_size=60 --num_pool_prompt_tokens=5 --num_prompts_per_task=20 --train_only_prompts --log_steps=10 --data_num=1000 --bleu_samples=100 --num_train_epochs=30 --patience=100 --warmup_steps=0 --save_last_checkpoints --always_save_model --project_name=debug_pool1 --name=1000_fnorm_proj_expansion --stream=concode_none,translate_java-cs,summarize_ruby,refine_small

# p4
CUDA_VISIBLE_DEVICES=0,1,2,3 python cont_gen.py --train_batch_size=16 --eval_batch_size=64 --prompt_pool --separate_projection --prompt_projection --projection_hid_dim=512 --projection_out_dim=64 --pool_lambda=0.5 --pool_size=60 --num_pool_prompt_tokens=5 --num_prompts_per_task=20 --train_only_prompts --log_steps=10 --data_num=1000 --bleu_samples=100 --num_train_epochs=30 --patience=100 --warmup_steps=0 --save_last_checkpoints --always_save_model --project_name=debug_pool1 --name=1000_sep_proj_contract --stream=concode_none,translate_java-cs,summarize_ruby,refine_small
CUDA_VISIBLE_DEVICES=4,5,6,7 python cont_gen.py --train_batch_size=16 --eval_batch_size=64 --prompt_pool --separate_projection --pool_freq_norm --prompt_projection --projection_hid_dim=512 --projection_out_dim=64 --pool_lambda=0.5 --pool_size=60 --num_pool_prompt_tokens=5 --num_prompts_per_task=20 --train_only_prompts --log_steps=10 --data_num=1000 --bleu_samples=100 --num_train_epochs=30 --patience=100 --warmup_steps=0 --save_last_checkpoints --always_save_model --project_name=debug_pool1 --name=1000_fnorm_sep_proj_contract --stream=concode_none,translate_java-cs,summarize_ruby,refine_small

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --train_batch_size=8 --eval_batch_size=32 --prompt_pool --separate_projection --pool_freq_norm --prompt_projection --projection_hid_dim=512 --projection_out_dim=64 --pool_lambda=0.5 --pool_size=80 --num_pool_prompt_tokens=5 --num_prompts_per_task=20 --train_only_prompts --log_steps=10 --data_num=1000 --bleu_samples=100 --num_train_epochs=30 --patience=100 --warmup_steps=0 --save_last_checkpoints --always_save_model --project_name=debug_pool1 --name=1000_pool80_fnorm_sep_proj_contract --stream=concode_none,translate_java-cs,summarize_ruby,refine_small
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --train_batch_size=8 --eval_batch_size=32 --prompt_pool --separate_projection --pool_freq_norm --prompt_projection --projection_hid_dim=512 --projection_out_dim=64 --pool_lambda=0.5 --pool_size=120 --num_pool_prompt_tokens=5 --num_prompts_per_task=20 --train_only_prompts --log_steps=10 --data_num=1000 --bleu_samples=100 --num_train_epochs=30 --patience=100 --warmup_steps=0 --save_last_checkpoints --always_save_model --project_name=debug_pool1 --name=1000_pool120_fnorm_sep_proj_contract --stream=concode_none,translate_java-cs,summarize_ruby,refine_small


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --train_batch_size=8 --eval_batch_size=32 --prompt_pool --separate_projection --pool_freq_norm --prompt_projection --projection_hid_dim=2048 --projection_out_dim=2048 --pool_lambda=0.5 --pool_size=120 --num_pool_prompt_tokens=5 --num_prompts_per_task=20 --train_only_prompts --log_steps=10 --data_num=1000 --bleu_samples=100 --num_train_epochs=30 --patience=100 --warmup_steps=0 --save_last_checkpoints --always_save_model --project_name=debug_pool1 --name=1000_pool120_fnorm_sep_proj_expansion --stream=concode_none,translate_java-cs,summarize_ruby,refine_small




CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python cont_gen.py --train_batch_size=16 --eval_batch_size=64 --prompt_pool --separate_projection --prompt_projection --projection_hid_dim=2048 --projection_out_dim=2048 --pool_lambda=0.5 --pool_size=60 --num_pool_prompt_tokens=5 --num_prompts_per_task=20 --train_only_prompts --log_steps=10 --data_num=1000 --bleu_samples=100 --num_train_epochs=30 --patience=100 --warmup_steps=0 --save_last_checkpoints --always_save_model --project_name=debug_pool1 --name=1000_sep_proj_expansion --stream=concode_none,translate_java-cs,summarize_ruby,refine_small
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --train_batch_size=16 --eval_batch_size=64 --prompt_pool --separate_projection --pool_freq_norm --prompt_projection --projection_hid_dim=2048 --projection_out_dim=2048 --pool_lambda=0.5 --pool_size=60 --num_pool_prompt_tokens=5 --num_prompts_per_task=20 --train_only_prompts --log_steps=10 --data_num=1000 --bleu_samples=100 --num_train_epochs=30 --patience=100 --warmup_steps=0 --save_last_checkpoints --always_save_model --project_name=debug_pool1 --name=1000_fnorm_sep_proj_expansion --stream=concode_none,translate_java-cs,summarize_ruby,refine_small




CUDA_VISIBLE_DEVICES=0,1,2,3 python cont_gen.py --train_batch_size=16 --eval_batch_size=100 --prompt_pool --pool_freq_norm --pool_lambda=0.5 --pool_size=80 --num_pool_prompt_tokens=5 --num_prompts_per_task=20 --train_only_prompts --log_steps=10 --data_num=500 --bleu_samples=50 --num_train_epochs=15 --patience=100 --warmup_steps=0 --save_last_checkpoints --always_save_model --project_name=debug_pool1 --name=500_fnorm_pool80 --stream=concode_none,translate_java-cs,summarize_ruby,refine_small
CUDA_VISIBLE_DEVICES=4,5,6,7 python cont_gen.py --train_batch_size=16 --eval_batch_size=100 --prompt_pool --pool_freq_norm --pool_lambda=0.5 --pool_size=100 --num_pool_prompt_tokens=5 --num_prompts_per_task=20 --train_only_prompts --log_steps=10 --data_num=500 --bleu_samples=50 --num_train_epochs=15 --patience=100 --warmup_steps=0 --save_last_checkpoints --always_save_model --project_name=debug_pool1 --name=500_fnorm_pool100 --stream=concode_none,translate_java-cs,summarize_ruby,refine_small


### Compute similarity
CUDA_VISIBLE_DEVICES=0 python compute_sim.py --name=500_fnorm_pool80~try=0 --eval_batch_size=80 --prompt_pool --num_prompts_per_task=20 --prompt_init=vocab --num_pool_prompt_tokens=5 --pool_size=80 --pool_freq --compute_avg_sim --data_num=500 --project_name=debug_pool  --stream=concode_none,translate_java-cs,summarize_ruby,refine_small


### PROMPT POOL: Cosine Similarity ###
### Works only on a single gpu. Inplace concatenation doesn't works hence DP losses the computed values.
CUDA_VISIBLE_DEVICES=0 python cont_gen.py --train_batch_size=8 --eval_batch_size=32 --compute_avg_sim --prompt_pool --pool_lambda=0.5 --pool_size=60 --num_pool_prompt_tokens=5 --num_prompts_per_task=20 --train_only_prompts --log_steps=10 --data_num=2000 --num_train_epochs=1 --warmup_steps=50 --save_last_checkpoints --always_save_model --project_name=debug_pool --name=query_sim --stream=concode_none,translate_java-cs,summarize_ruby,refine_small --no_eval --no_eval_bleu --no_eval_all --no_test
CUDA_VISIBLE_DEVICES=1 python cont_gen.py --train_batch_size=8 --eval_batch_size=32 --compute_avg_sim --prompt_pool --prompt_projection --projection_out_dim=64 --pool_lambda=0.5 --pool_size=60 --num_pool_prompt_tokens=5 --num_prompts_per_task=20 --train_only_prompts --log_steps=10 --data_num=2000 --num_train_epochs=1 --warmup_steps=50 --save_last_checkpoints --always_save_model --project_name=debug_pool --name=query_project_sim --stream=concode_none,translate_java-cs,summarize_ruby,refine_small --no_eval --no_eval_bleu --no_eval_all --no_test


### PROMPT POOL: Prompt frequency ###
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --train_batch_size=8 --eval_batch_size=32 --pool_freq --prompt_pool --pool_lambda=0.5 --pool_size=60 --num_pool_prompt_tokens=5 --num_prompts_per_task=20 --train_only_prompts --log_steps=10 --data_num=1000 --warmup_steps=50 --save_last_checkpoints --always_save_model --project_name=debug_pool --name=freq_dist --stream=concode_none,translate_java-cs,summarize_ruby,refine_small --no_eval_all --no_test
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --train_batch_size=8 --eval_batch_size=32 --pool_freq_norm --prompt_pool --pool_lambda=0.5 --pool_size=60 --num_pool_prompt_tokens=5 --num_prompts_per_task=20 --train_only_prompts --log_steps=10 --data_num=1000 --warmup_steps=50 --save_last_checkpoints --always_save_model --project_name=debug_pool --name=1k_freq_dist_nonorn --stream=concode_none,translate_java-cs,summarize_ruby,refine_small --no_eval_all --no_test



### Debugging Prompt Pool ###
CUDA_VISIBLE_DEVICES=0,1,2,3 python cont_gen.py --train_batch_size=8 --eval_batch_size=32 --prompt_pool --pool_lambda=0.5 --pool_size=60 --num_pool_prompt_tokens=5 --num_prompts_per_task=20 --train_only_prompts --log_steps=10 --data_num=2000 --warmup_steps=50 --save_last_checkpoints --always_save_model --project_name=debug_pool --name=CSTR_pool --stream=concode_none,translate_java-cs,summarize_ruby,refine_small
CUDA_VISIBLE_DEVICES=4,5,6,7 python cont_gen.py --train_batch_size=8 --eval_batch_size=32 --prompt_projection --projection_hid_dim=512 --projection_out_dim=64 --prompt_pool --pool_lambda=0.5 --pool_size=60 --num_pool_prompt_tokens=5 --num_prompts_per_task=20 --train_only_prompts --log_steps=10 --data_num=2000 --warmup_steps=50 --save_last_checkpoints --always_save_model --project_name=debug_pool --name=CSTR_pool_project_down --stream=concode_none,translate_java-cs,summarize_ruby,refine_small
CUDA_VISIBLE_DEVICES=0,1,2,3 python cont_gen.py --train_batch_size=8 --eval_batch_size=32 --prompt_projection --projection_hid_dim=1024 --projection_out_dim=1024 --prompt_pool --pool_lambda=0.5 --pool_size=60 --num_pool_prompt_tokens=5 --num_prompts_per_task=20 --train_only_prompts --log_steps=10 --data_num=2000 --warmup_steps=50 --save_last_checkpoints --always_save_model --project_name=debug_pool --name=CSTR_pool_project_up --stream=concode_none,translate_java-cs,summarize_ruby,refine_small

CUDA_VISIBLE_DEVICES=4,5,6,7 python cont_gen.py --train_batch_size=8 --eval_batch_size=32 --pool_freq_norm --prompt_pool --pool_lambda=0.5 --pool_size=60 --num_pool_prompt_tokens=5 --num_prompts_per_task=20 --train_only_prompts --log_steps=10 --data_num=2000 --warmup_steps=50 --save_last_checkpoints --always_save_model --project_name=debug_pool --name=CSTR_pool_freq --stream=concode_none,translate_java-cs,summarize_ruby,refine_small
CUDA_VISIBLE_DEVICES=0,1,2,3 python cont_gen.py --train_batch_size=8 --eval_batch_size=32 --pool_freq_norm --prompt_projection --projection_hid_dim=512 --projection_out_dim=64 --prompt_pool --pool_lambda=0.5 --pool_size=60 --num_pool_prompt_tokens=5 --num_prompts_per_task=20 --train_only_prompts --log_steps=10 --data_num=2000 --warmup_steps=50 --save_last_checkpoints --always_save_model --project_name=debug_pool --name=CSTR_pool_freq_project_down --stream=concode_none,translate_java-cs,summarize_ruby,refine_small
CUDA_VISIBLE_DEVICES=4,5,6,7 python cont_gen.py --train_batch_size=8 --eval_batch_size=32 --pool_freq_norm --prompt_projection --projection_hid_dim=1024 --projection_out_dim=1024 --prompt_pool --pool_lambda=0.5 --pool_size=60 --num_pool_prompt_tokens=5 --num_prompts_per_task=20 --train_only_prompts --log_steps=10 --data_num=2000 --warmup_steps=50 --save_last_checkpoints --always_save_model --project_name=debug_pool --name=CSTR_pool_freq_project_up --stream=concode_none,translate_java-cs,summarize_ruby,refine_small


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --train_batch_size=8 --eval_batch_size=32 --pool_freq_norm --prompt_pool --pool_lambda=0.5 --pool_size=60 --num_pool_prompt_tokens=5 --num_prompts_per_task=20 --train_only_prompts --log_steps=10 --data_num=2000 --warmup_steps=50 --save_last_checkpoints --always_save_model --project_name=debug_pool --name=CSTR_pool --stream=concode_none,translate_java-cs,summarize_ruby,refine_small



######## TUNING PROMPT POOLING ###########
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --bleu_samples=500 --num_train_epochs=10 --log_steps=10 --data_num=5000 --warmup_steps=1 --save_last_checkpoints --always_save_model --project_name=debug_pool --name=naive --stream=concode_none,translate_java-cs,summarize_ruby,refine_small
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --train_batch_size=8 --eval_batch_size=32 --bleu_samples=500 --num_train_epochs=10 --prompt_pool --pool_lambda=0.5 --pool_size=60 --num_pool_prompt_tokens=5 --num_prompts_per_task=20 --train_only_prompts --log_steps=10 --data_num=5000 --warmup_steps=1 --save_last_checkpoints --always_save_model --project_name=debug_pool --name=pool_60_20_5 --stream=concode_none,translate_java-cs,summarize_ruby,refine_small
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --train_batch_size=8 --eval_batch_size=32 --bleu_samples=500 --num_train_epochs=10 --prompt_pool --pool_lambda=0.5 --pool_size=40 --num_pool_prompt_tokens=5 --num_prompts_per_task=20 --train_only_prompts --log_steps=10 --data_num=5000 --warmup_steps=1 --save_last_checkpoints --always_save_model --project_name=debug_pool --name=pool_40_20_5 --stream=concode_none,translate_java-cs,summarize_ruby,refine_small
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --train_batch_size=8 --eval_batch_size=32 --bleu_samples=500 --num_train_epochs=10 --prompt_pool --pool_lambda=0.5 --pool_size=80 --num_pool_prompt_tokens=5 --num_prompts_per_task=20 --train_only_prompts --log_steps=10 --data_num=5000 --warmup_steps=1 --save_last_checkpoints --always_save_model --project_name=debug_pool --name=pool_80_20_5 --stream=concode_none,translate_java-cs,summarize_ruby,refine_small
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --train_batch_size=8 --eval_batch_size=32 --bleu_samples=500 --num_train_epochs=10 --prompt_pool --pool_lambda=0.5 --pool_size=100 --num_pool_prompt_tokens=5 --num_prompts_per_task=20 --train_only_prompts --log_steps=10 --data_num=5000 --warmup_steps=1 --save_last_checkpoints --always_save_model --project_name=debug_pool --name=pool_100_20_5 --stream=concode_none,translate_java-cs,summarize_ruby,refine_small



# ##### tuning prompt tuning ####
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --prompt_lr=0.1 --num_prompts_per_task=20 --train_only_prompts --log_steps=10 --data_num=-1 --warmup_steps=1000 --save_last_checkpoints --always_save_model --project_name=tune_prompt_params --name=np20_lr0.1_CSTR_onlyprompt --stream=concode_none,translate_java-cs,summarize_ruby,refine_small
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --prompt_lr=1 --num_prompts_per_task=20 --train_only_prompts --log_steps=10 --data_num=-1 --warmup_steps=1000 --save_last_checkpoints --always_save_model --project_name=tune_prompt_params --name=np20_lr1_CSTR_onlyprompt --stream=concode_none,translate_java-cs,summarize_ruby,refine_small
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --prompt_lr=10 --num_prompts_per_task=20 --train_only_prompts --log_steps=10 --data_num=-1 --warmup_steps=1000 --save_last_checkpoints --always_save_model --project_name=tune_prompt_params --name=np20_lr10_CSTR_onlyprompt --stream=concode_none,translate_java-cs,summarize_ruby,refine_small
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --prompt_lr=100 --num_prompts_per_task=20 --train_only_prompts --log_steps=10 --data_num=-1 --warmup_steps=1000 --save_last_checkpoints --always_save_model --project_name=tune_prompt_params --name=np20_lr100_CSTR_onlyprompt --stream=concode_none,translate_java-cs,summarize_ruby,refine_small

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --train_batch_size=8 --eval_batch_size=32 --num_prompts_per_task=50 --train_only_prompts --log_steps=10 --data_num=-1 --warmup_steps=1000 --save_last_checkpoints --always_save_model --project_name=tune_prompt_params --name=np50_lr100_CSTR_onlyprompt --stream=concode_none,translate_java-cs,summarize_ruby,refine_small
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --train_batch_size=8 --eval_batch_size=32 --num_prompts_per_task=100 --train_only_prompts --log_steps=10 --data_num=-1 --warmup_steps=1000 --save_last_checkpoints --always_save_model --project_name=tune_prompt_params --name=np100_lr100_CSTR_onlyprompt --stream=concode_none,translate_java-cs,summarize_ruby,refine_small

# ##### debug prompts ###
# CUDA_VISIBLE_DEVICES=0,1,2,3 python cont_gen.py --bleu_samples=500 --num_train_epochs=5 --log_steps=10 --data_num=5000 --warmup_steps=100 --save_last_checkpoints --always_save_model --project_name=debug_prompt --name=ruby --stream=summarize_ruby
# CUDA_VISIBLE_DEVICES=4,5,6,7 python cont_gen.py --bleu_samples=100 --num_train_epochs=5 --num_prompts_per_task=30 --train_only_prompts --log_steps=10 --data_num=5000 --warmup_steps=100 --save_last_checkpoints --always_save_model --project_name=debug_prompt --name=ruby_onlyprompt --stream=summarize_ruby
# CUDA_VISIBLE_DEVICES=4,5,6,7 python cont_gen.py --bleu_samples=100 --num_train_epochs=5 --num_prompts_per_task=30 --log_steps=10 --data_num=5000 --warmup_steps=100 --save_last_checkpoints --always_save_model --project_name=debug_prompt --name=ruby_onlyprompt --stream=summarize_ruby
# CUDA_VISIBLE_DEVICES=0 python cont_gen.py --no_eval --no_eval_bleu --no_eval_all --no_test --num_train_epochs=500 --num_prompts_per_task=30 --train_only_prompts --log_steps=10 --data_num=8 --warmup_steps=1 --project_name=debug_prompt --name=overfit_ruby_onlyprompt --stream=summarize_ruby



### CL after Starting from a T5 model.
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python cont_gen.py --log_steps=10 --data_num=-1 --full_matrix_eval --model_name_or_path=t5-small --warmup_steps=1000 --save_last_checkpoints --always_save_model --project_name=summ_exp3 --name=t5_all --stream=summarize

### Individual Tasks
CUDA_VISIBLE_DEVICES=0 python cont_gen.py --log_steps=10 --data_num=-1 --warmup_steps=1000 --save_last_checkpoints --always_save_model --project_name=task_exp2 --name=concode --task=concode_none

### Zeroshot Eval
CUDA_VISIBLE_DEVICES=7 python cont_gen.py --zeroshot --stream=summarize --name=zeroshot --project_name=summ_exp3 --debug --eval_batch_size=90 --no_train --no_eval --no_eval_bleu --no_eval_all --log_steps=10 --data_num=-1



# kill all processes on gpu XXX
for i in $(sudo lsof /dev/nvidiaXXX | grep python  | awk '{print $2}' | sort -u); do kill -9 $i; done
