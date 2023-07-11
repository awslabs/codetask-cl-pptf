
# Continual Learning for Seq2seq models for Code related tasks.

This project contains the code for Continual Learning for Seq2seq models for Code related tasks.


#### Relevant Repositories
This codebase has some code or ideas ported from the following repositories.
1. [CodeXGLUE](https://github.com/microsoft/CodeXGLUE)
2. [CodeT5](https://github.com/salesforce/CodeT5)
3. [Learning to Prompt](https://github.com/google-research/l2p)

#### Creating an environment
Use the file `src/environment.yml` to create a conda environment. The following command can be used `conda env create --file=environment.yml`.

#### Folder Structure
```
src
│   dataloaders	# dataloader for CL tasks.
│   evaluator	# Original from CodeT5 for evaluation.
│	plots	# File to make some basic plots related to similarity and prompt frequency.
|	tokenizer	# Original from CodeT5 for some tokenization. Not used for us.
└───sh
|	|	final_runs.sh	# Contains commands for main experiments. These can be used as examples to run the code. For more info on the arguments please look at the config.py file.
└───models
|	|	T5prompt.py
└───utils
│   │   metrics.py	# Main file which implements the metrics.
│   │   replay.py	# Main file to implement replay buffer.
│   │	configs.py	# argparse arguments, etc
│   cont_gen.py	# Main file for running CL experiments.
|	analyse.ipynb	# Main file for analysing the query-key matching analysis.
|	run_gen.py	# Original file from codeT5 to finetune on a single file.
|	run_multi_gen.py	# Modified file from CodeT5 to run multitask learning. Has some hacks to get it works for us.
```

#### Sample commands
1. Basic Prompt Pooling: `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --keys_agg=random --pool_freq --name=pool100 --pool_size=100 --prompt_method=pool --num_prompts_per_task=20 --train_only_prompts --bleu_samples=5000 --warmup_steps=500 --train_batch_size=8 --eval_batch_size=32 --log_steps=10 --data_num=-1 --save_last_checkpoints --always_save_model --project_name=final_1 --stream=concode_none,translate_java-cs,summarize_ruby,refine_small`

2. PP + ER: `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --replay=ring --buffer_size=500 --buffer_bs=2 --keys_agg=random --pool_freq --name=pool100_ER500 --pool_size=100 --prompt_method=pool --num_prompts_per_task=20 --train_only_prompts --bleu_samples=5000 --warmup_steps=500 --train_batch_size=8 --eval_batch_size=32 --log_steps=10 --data_num=-1 --save_last_checkpoints --always_save_model --project_name=final_1 --stream=concode_none,translate_java-cs,summarize_ruby,refine_small`

3. PP + TF + ER: `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --pool_teacher --num_shared_keys_per_pair=2 --replay=ring --buffer_size=500 --buffer_bs=2 --keys_agg=random --pool_freq --name=pool100_teacher_ER500 --pool_size=100 --prompt_method=pool --num_prompts_per_task=20 --train_only_prompts --bleu_samples=5000 --warmup_steps=500 --train_batch_size=16 --eval_batch_size=64 --log_steps=10 --data_num=-1 --save_last_checkpoints --always_save_model --project_name=teacher_tune --stream=concode_none,translate_java-cs,summarize_ruby,refine_small`

4. PP + Fixed Assignment: `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --pool_freq --prompt_lr=100 --name=pool200_fixed_selection_plr100 --pool_size=200 --prompt_method=pool_fixed --num_pool_prompt_tokens=5 --num_prompts_per_task=20 --train_only_prompts --bleu_samples=5000 --warmup_steps=500 --train_batch_size=8 --eval_batch_size=32 --log_steps=10 --data_num=-1 --save_last_checkpoints --always_save_model --project_name=final_1 --stream=concode_none,translate_java-cs,summarize_ruby,refine_small`

5. ShPT: `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --prompt_lr=100 --name=shpt --prompt_method=shpt --num_prompts_per_task=100 --train_only_prompts --bleu_samples=5000 --warmup_steps=500 --train_batch_size=8 --eval_batch_size=32 --log_steps=10 --data_num=-1 --save_last_checkpoints --always_save_model --project_name=final_1 --stream=concode_none,translate_java-cs,summarize_ruby,refine_small`

6. TSPT: `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --name=TSPT --prompt_method=tspt --prompt_lr=100 --num_prompts_per_task=100 --train_only_prompts --bleu_samples=5000 --warmup_steps=500 --train_batch_size=8 --eval_batch_size=32 --log_steps=10 --data_num=-1 --save_last_checkpoints --always_save_model --project_name=final_1 --stream=concode_none,translate_java-cs,summarize_ruby,refine_small`

7. NSL + ER: `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --replay=ring --buffer_size=500 --add_task_prefix --add_lang_ids --bleu_samples=5000 --log_steps=10 --data_num=-1 --warmup_steps=500 --save_last_checkpoints --always_save_model --project_name=final_1 --name=nsl_ER500 --stream=concode_none,translate_java-cs,summarize_ruby,refine_small`

8. Summ + NSL + ER: `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cont_gen.py --name=nsl_ER500 --replay=ring --buffer_size=500 --add_task_prefix --add_lang_ids --bleu_samples=5000 --log_steps=10 --data_num=-1 --warmup_steps=100 --train_batch_size=8 --eval_batch_size=32 --save_last_checkpoints --always_save_model --project_name=summ_pool --stream=summarize`