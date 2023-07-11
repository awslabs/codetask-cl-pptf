CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python cont_gen.py --name=TSPT --prompt_method=tspt --prompt_lr=100 --num_prompts_per_task=100 --train_only_prompts --bleu_samples=5000 --warmup_steps=500 --train_batch_size=8 --eval_batch_size=32 --log_steps=10 --data_num=-1 --save_last_checkpoints --always_save_model --project_name=final_1 --stream=concode_none,translate_java-cs,summarize_ruby,refine_small

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python cont_gen.py --pool_freq --prompt_lr=1 --name=pool200_fixed_selection_plr1 --pool_size=200 --prompt_method=pool_fixed --num_pool_prompt_tokens=5 --num_prompts_per_task=20 --train_only_prompts --bleu_samples=5000 --warmup_steps=500 --train_batch_size=8 --eval_batch_size=32 --log_steps=10 --data_num=-1 --save_last_checkpoints --always_save_model --project_name=final_1 --stream=concode_none,translate_java-cs,summarize_ruby,refine_small