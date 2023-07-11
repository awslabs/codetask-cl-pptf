# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import random
import torch
import logging
import multiprocessing
import numpy as np
import os

logger = logging.getLogger(__name__)


def add_args(parser):
    parser.add_argument("--task", type=str, default=None, help="")
    parser.add_argument("--sub_task", type=str, default='')
    parser.add_argument("--stream", type=str, default=None)

    parser.add_argument("--lang", type=str, default='')
    parser.add_argument("--eval_task", type=str, default='')
    parser.add_argument("--model_type", default="codet5", type=str, choices=['roberta', 'bart', 'codet5'])
    parser.add_argument("--add_lang_ids", action='store_true')
    parser.add_argument("--data_num", default=-1, type=int)
    parser.add_argument("--bleu_samples", default=5000, type=int)

    # task specific params
    parser.add_argument('--patience', nargs='+', default=None, type=int, help='Patience for early stopping.')
    parser.add_argument('--num_train_epochs', nargs='+', default=None, type=int, help='Number of epochs')
    parser.add_argument('--learning_rate', nargs='+', default=None, type=float, help='Learning rate')
    parser.add_argument('--max_source_length', nargs='+', default=None, type=int, help='max src len')
    parser.add_argument('--max_target_length', nargs='+', default=None, type=int, help='max tgt len')
    parser.add_argument('--train_batch_size', nargs='+', default=None, type=int, help='Batch size per GPU/CPU for training.')
    parser.add_argument('--eval_batch_size', nargs='+', default=None, type=int, help='Batch size per GPU/CPU for evaluation.')

    ## replay params
    parser.add_argument("--replay", default='res', type=str, help="wheather to use replay or not")
    parser.add_argument("--buffer_size", default=0, type=int, help="total size of the buffer across all tasks")
    parser.add_argument("--buffer_bs", default=8, type=int, help="batch size to use for buffer.")
    parser.add_argument("--replay_epoch_end", action='store_true')
    parser.add_argument("--alpha", default=0, type=float, help="Mixing weight for buffer loss.")

    ## Prompting params
    parser.add_argument("--pool_lambda", default=0.1, type=float, help="mixing weight for pompt pool loss and mle")
    parser.add_argument("--center_lambda", default=0.1, type=float, help="the weight for cluster center loss if applicable.")
    parser.add_argument("--prompt_loss_type", default='', type=str, help="to add other types of loss, say cluster_cen loss")
    parser.add_argument("--pool_teacher", default='', type=str, help="To use teacher forcing during with prompt pooling")
    parser.add_argument("--num_shared_keys_per_pair", default=2, type=int, help="number of shared keys between each pair of prompts.")
    parser.add_argument("--num_pool_prompt_tokens", default=5, type=int, help="How many prompts corresponding to each key.")
    parser.add_argument("--num_prompts_per_task", default=0, type=int, help="number of keys to use for each task.")
    parser.add_argument("--pool_size", default=60, type=int, help="total size of the pool, this is not used when we have teacher forcing.")
    parser.add_argument("--task_specific_ratio", default=0, type=float, help="the number of fixed prompts when asssining a fraction of prompts to patricular tasks.")

    parser.add_argument("--query_pooling_mode", default='mean', type=str)
    parser.add_argument("--keys_agg", default='random', type=str)
    parser.add_argument("--prompt_lr", default=10, type=float)
    parser.add_argument("--query_key_lr", default=0.1, type=float)

    parser.add_argument("--train_only_prompts", action='store_true', help="add this to freeze all parameters apart from key and prompts")
    parser.add_argument("--no_keys", action='store_true')
    parser.add_argument("--prompt_method", default="", type=str)
    parser.add_argument("--io_queries", action='store_true')
    parser.add_argument("--batched_prompts", action='store_true')

    parser.add_argument("--pool_freq_norm", action='store_true', help="to perform frequency normalization")
    parser.add_argument("--pool_freq", action='store_true', help="to count frequency")
    parser.add_argument("--compute_avg_sim", action='store_true')

    parser.add_argument("--uniform_scale", default=0.001, type=float)
    parser.add_argument("--prompt_init", default='data', type=str, help="initilization method for prompts")
    parser.add_argument("--prompt_key_init", default='data', type=str, help="initilization method for keys.")

    parser.add_argument("--prompt_projection", action='store_true')
    parser.add_argument("--separate_projection", action='store_true')
    parser.add_argument("--projection_hid_dim", default=512, type=int)
    parser.add_argument("--projection_out_dim", default=512, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--projection_plot", default='', type=str)


    ## Directories.
    parser.add_argument("--project_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default='data',)
    parser.add_argument("--output_dir", default='saved_runs', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cache_path", type=str, default='data/cache',)
    parser.add_argument("--summary_dir", type=str, default='saved_runs/logs',)
    parser.add_argument("--res_dir", type=str, default='',)
    parser.add_argument("--res_fn", type=str, default='')

    parser.add_argument("--add_task_prefix", action='store_true', help="Whether to add task prefix for t5 and codet5")
    parser.add_argument("--save_last_checkpoints", action='store_true')
    parser.add_argument("--always_save_model", action='store_true')
    parser.add_argument("--calc_stats", action='store_true')

    # wandb params
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--name", type=str, default='test')
    parser.add_argument("--project_name", type=str, default='debug')


    ## Huggingface params.
    parser.add_argument("--config_name", default="Salesforce/codet5-small", type=str, help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--model_name_or_path", default="Salesforce/codet5-small", type=str, help="Path to pre-trained model: e.g. Salesforce/codet5-small")
    parser.add_argument("--tokenizer_name", default="Salesforce/codet5-small", type=str, help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--load_model_path", default=None, type=str, help="Path to trained model: Should contain the .bin files")

    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str,
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str,
                        help="The test filename. Should contain the .jsonl files for this task.")


    parser.add_argument("--no_train", action='store_true', help="Whether to run eval on the train set.")
    parser.add_argument("--no_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--no_eval_bleu", action='store_true', help="Whether to evaluate bleu on dev set.")
    parser.add_argument("--no_eval_all", action='store_true', help="Whether to run eval on all tasks dev set after each task.")
    parser.add_argument("--no_test", action='store_true', help="Whether to evaluate on test set.")
    parser.add_argument("--full_matrix_eval", action='store_true', help="evaluate on future tasks as well in each epoch.")
    parser.add_argument("--zeroshot", action='store_true', help="Evaluate zeroshot performance on the test set.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")

    parser.add_argument("--adafactor", action='store_true', help="Use adafactor instead of AdamW")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--beam_size", default=5, type=int, help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument("--num_workers", default=os.cpu_count(), type=int, )
    parser.add_argument("--pin_memory", default=True, type=bool, )
    parser.add_argument("--save_steps", default=-1, type=int, )
    parser.add_argument("--num_saves", default=10, type=int, )
    parser.add_argument("--log_steps", default=-1, type=int, )
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int, help="")
    parser.add_argument("--train_steps", default=-1, type=int, help="")
    parser.add_argument("--warmup_steps", default=100, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=1234, help="random seed for initialization")

    args = parser.parse_args()

    if args.task in ['summarize']:
        args.lang = args.sub_task
    elif args.task in ['refine', 'concode', 'clone']:
        args.lang = 'java'
    elif args.task == 'defect':
        args.lang = 'c'
    elif args.task == 'translate':
        args.lang = 'c_sharp' if args.sub_task == 'java-cs' else 'java'

    args.project_name = f"aws-{args.project_name}"
    args.cpu_cont = multiprocessing.cpu_count()

    args.data_dir = os.path.join(args.project_dir, args.data_dir)
    args.cache_path = os.path.join(args.project_dir, args.cache_path)
    args.output_dir = os.path.join(args.project_dir, args.output_dir)
    os.makedirs(args.cache_path, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    i = 0

    if not args.compute_avg_sim:
        while True:
            args.run_output_dir = f"{args.output_dir}/{args.project_name}/{args.name}~try={i}"
            if not os.path.exists(args.run_output_dir):
                os.makedirs(args.run_output_dir)
                args.name = args.name + f"~try={i}"
                break
            i += 1
    else:
        args.run_output_dir = f"{args.output_dir}/{args.project_name}/{args.name}"

    args.log_dir = os.path.join(args.run_output_dir, 'logs')
    args.checkpoint_dir = os.path.join(args.run_output_dir, 'checkpoints')
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    return args


def set_dist(args):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        # Setup for distributed data parallel
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    cpu_cont = multiprocessing.cpu_count()
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, cpu count: %d",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), cpu_cont)
    args.device = device
    args.cpu_cont = cpu_cont

def get_task_arglist(args, keys, datamodule):
    for key in keys:
        if hasattr(args, key):
            att_value = getattr(args, key)
            new_att_value = []
        else:
            raise ValueError(f"Key {key} is not in args!")

        if att_value is None:
            for task in datamodule.all_tasks:
                key_map = {"num_train_epochs": 'epoch', 'learning_rate': 'lr', 'patience': 'patience', 'max_source_length':'src_len',
                            'max_target_length':'trg_len', 'train_batch_size':'tbs', 'eval_batch_size':'ebs'}
                task_val = datamodule.task_params[task][key_map[key]]
                new_att_value.append(task_val)
        elif len(att_value) == 1:
            new_att_value = att_value * len(datamodule.all_tasks)
        elif len(att_value) == len(datamodule.all_tasks):
            new_att_value = att_value
        setattr(args, key, new_att_value)
    return args

def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
