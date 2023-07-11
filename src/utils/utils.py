# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from torch.utils.data import TensorDataset
import numpy as np
import wandb
from pathlib import Path
import logging
import os
import random
import torch
import torch.nn as nn
import time
from tqdm import tqdm
from collections import defaultdict as ddict
from utils._utils import *
from transformers import AdamW, Adafactor, get_linear_schedule_with_warmup



logger = logging.getLogger(__name__)


def checkpoint_and_earlystop(args, results, best_metric, metric_key, model, filehandle, not_dec_cnt, cur_epoch, curr_task, mode='max', patience=None):
    early_stop = False
    if metric_key == 'bleu_em':
        curr_metric = results['bleu_em']
        dev_bleu = results['bleu']
        dev_em = results['em']
    elif 'ppl' in metric_key:
        curr_metric = results[metric_key]

    if mode == 'max':
        better = curr_metric > best_metric
    elif mode == 'min':
        better = curr_metric < best_metric
    else:
        raise ValueError()

    if better:
        not_dec_cnt[metric_key] = 0
        if metric_key == 'bleu_em':
            msg = f"[Task:{curr_task} Epoch:{cur_epoch}] Best bleu+em:  {best_metric} ---> {curr_metric:.2f} (bleu: {dev_bleu:.2f}, em: {dev_em:.2f})\n".upper()
            logger.info(msg)
            filehandle.write(msg)
        else:
            msg = f"[Task:{curr_task} Epoch:{cur_epoch}] Best {metric_key} changed from {best_metric:.4f} ---> {curr_metric:.4f}\n".upper()
            logger.info(msg)
            filehandle.write(msg)
        best_metric = curr_metric
        # Save best checkpoint for best bleu
        check_name = 'best-bleu' if 'bleu' in metric_key else 'best-ppl'
        output_dir = os.path.join(args.run_output_dir, 'checkpoints')
        os.makedirs(output_dir, exist_ok=True)
        if args.data_num == -1 or args.always_save_model:
            model_to_save = model.module if hasattr(model, 'module') else model
            output_model_file = os.path.join(output_dir, f"{check_name}.bin")
            torch.save(model_to_save.state_dict(), output_model_file)
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, f"{curr_task}.bin"))
            logger.info(f"Save the best {metric_key} model into {output_model_file}")
    else:
        not_dec_cnt[metric_key] += 1
        logger.info(f"{metric_key} did not improve for {not_dec_cnt[metric_key]} epochs")
        if 'bleu' in metric_key:
            msg = f"[Task:{curr_task} Epoch:{cur_epoch}] Best bleu+em: did not drop for {not_dec_cnt[metric_key]} epochs (bleu: {dev_bleu:.2f}, em: {dev_em:.2f})\n"
            logger.info(msg)
            filehandle.write(msg)
        else:
            msg = f"[Task:{curr_task} Epoch:{cur_epoch}] Best {metric_key} did not drop for {not_dec_cnt[metric_key]} epochs.\n"
            logger.info(msg)
            filehandle.write(msg)

        pat = patience if patience is not None else args.patience[0]
        if all([x > pat for x in not_dec_cnt.values()]):
            stop_early_str = f"[Task:{curr_task} Epoch:{cur_epoch}] Early stop as " + f", ".join(f"{k}={v}" for k,v in not_dec_cnt.items())
            logger.info(stop_early_str.upper())
            filehandle.write(stop_early_str.upper())
            logger.info("  " + "*" * 20)
            early_stop = True
    return best_metric, early_stop, not_dec_cnt

def freeze_code_model_parameters(model):
    for n, p in model.named_parameters():
        if "code_model" in n:
            p.requires_grad = False

def freeze_query_base_model_parameters(model):
    for n, p in model.named_parameters():
        if "query_base_model" in n:
            p.requires_grad = False

def get_optimizer(args, model, dataloader, lr=None, warmup_steps=None, epochs=None):
    freeze_query_base_model_parameters(model)

    optimizer_grouped_parameters = [ {'params': [p for n, p in model.named_parameters() if ('prompt' in n)], 'weight_decay': 0.0, 'lr': args.prompt_lr} ]
    if not args.no_keys:
        optimizer_grouped_parameters.append({'params': [p for n, p in model.named_parameters() if 'pool_keys' in n], 'weight_decay': 0.0, 'lr': args.query_key_lr})
    optimizer_grouped_parameters.append({'params': [p for n, p in model.named_parameters() if 'query_proj' in n], 'weight_decay': 0.0, 'lr': args.query_key_lr})

    trainable_params = [n for n, p in model.named_parameters() if (('prompt' in n) and p.requires_grad)]
    if not args.no_keys:
        trainable_params += [n for n, p in model.named_parameters() if ('pool_keys' in n and p.requires_grad)]
    trainable_params += [n for n, p in model.named_parameters() if ('query_proj' in n and p.requires_grad)]

    if args.train_only_prompts:
        print('\nTraining only prompt parameters\n'.upper())
        freeze_code_model_parameters(model)
    else:
        if args.prompt_method == "":
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters()], 'weight_decay': 0.0, 'lr': lr},
            ]
            trainable_params += [n for n, p in model.named_parameters()]
        else:
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if 'code_model' in n], 'weight_decay': 0.0, 'lr': lr},
            ]
            trainable_params += [n for n, p in model.named_parameters() if 'code_model' in n]

    print(f"Trainable Parameters: {trainable_params}")
    optimizer_cls = Adafactor if args.adafactor else AdamW
    if args.adafactor:
        optimizer_cls = Adafactor
        optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
    else:
        optimizer_cls = AdamW
        optimizer_kwargs = {
            "eps": args.adam_epsilon,
        }
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    num_train_optimization_steps = epochs * len(dataloader)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=num_train_optimization_steps)
    return optimizer, scheduler


def load_and_cache_gen_data(args, task, sub_task, filename, pool, tokenizer, split_tag, max_source_length, max_target_length, only_src=False, is_sample=False, curr_task=None, bleu_samples=5000):
    # cache the data into args.cache_path except it is sampled
    # only_src: control whether to return only source ids for bleu evaluating (dev/test)
    # return: examples (Example object), data (TensorDataset)
    data_tag = 'datanum_all' if args.data_num == -1 else f'datanum_{args.data_num}'
    os.makedirs(f'{args.cache_path}/{task}', exist_ok=True)
    # if curr_task is not None:
    #     savename = f"{curr_task}_{savename}"
    param_names = ['task', 'subtask', 'model', 'tokenizer', 'msl', 'mtl', 'task_prefix', 'lang_ids']
    params = [task, sub_task, args.model_name_or_path, args.tokenizer_name, max_source_length, max_target_length, args.add_task_prefix, args.add_lang_ids]
    assert len(params) == len(param_names)
    identifier = "-".join(f'{param_names[i]}_{params[i]}' for i in range(len(params)))
    savename = f"{split_tag}{'_src' if only_src else ''}_{identifier}_{data_tag}".replace('/', '_')
    cache_fn = f'{args.cache_path}/{task}/{savename}.pt'
    examples = read_examples(filename, args.data_num, task)

    if is_sample:
        examples = random.sample(examples, min(bleu_samples, len(examples)))

    if args.calc_stats:
        if split_tag == 'train':
            calc_stats(examples, tokenizer, is_tokenize=True)
        else:
            calc_stats(examples)

    if os.path.exists(cache_fn) and not is_sample and args.debug:
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        if is_sample:
            logger.info(f"Sample {min(bleu_samples, len(examples))} data for computing bleu from {filename}")
        else:
            logger.info("Create cache data into %s", cache_fn)
        tuple_examples = [(example, idx, tokenizer, args, split_tag, task, sub_task, max_source_length, max_target_length) for idx, example in enumerate(examples)]
        features = pool.map(convert_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        # if split_tag == 'test' or only_src:
        #     data = TensorDataset(all_source_ids)
        # else:
        all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
        data = TensorDataset(all_source_ids, all_target_ids)
        if args.local_rank in [-1, 0] and not is_sample:
            torch.save(data, cache_fn)
    return examples, data

def get_reps_cosine_similarity(model):
    ind_model = model.module if isinstance(model, nn.DataParallel) else model
    task_reps = []
    for name, buf in ind_model._buffers.items():
        if 'reps' not in name:
            continue
        task_reps.append(torch.nn.functional.normalize(buf, p=2, dim=1))
    task_avg_sim = torch.empty((len(task_reps), len(task_reps)))
    for i, treps_i in enumerate(task_reps):
        for j, treps_j in enumerate(task_reps):
            task_avg_sim[i,j] = torch.tensordot(treps_i, treps_j, dims=([1],[1])).mean()
    return task_avg_sim


def getlang(task):
    t, s = task.split('_')
    if t in ['summarize']:
        lang = s
    elif t in ['refine', 'concode', 'clone']:
        lang = 'java'
    elif t == 'defect':
        lang = 'c'
    elif t == 'translate':
        lang = 'c_sharp' if s == 'java-cs' else 'java'
    return lang


def get_subtasks(task, all=False):
    if task == 'summarize':
        if all:
            sub_tasks = ['ruby', 'javascript', 'go', 'python', 'java', 'php']
        else:
            sub_tasks = ['ruby']
    elif task == 'translate':
        if all:
            sub_tasks = ['java-cs', 'cs-java']
        else:
            sub_tasks = ['java-cs']
    elif task == 'refine':
        if all:
            sub_tasks = ['small', 'medium']
        else:
            sub_tasks = ['small']
    else:
        sub_tasks = ['none']
    return sub_tasks


def get_tasks_mtl(task_names=None, sub_tasks=None):
    if task_names is not None and len(task_names) == 1 and task_names[0] != "all_tasks":
        task_list = task_names
        sub_tasks = get_subtasks(task_list[0]) if sub_tasks is None else sub_tasks
        pairs = [(task_list[0], st) for st in sub_tasks]
    elif len(task_names) == 1 and task_names[0] == 'all_tasks':
        task_list = ['summarize', 'translate', 'refine', 'concode']
        pairs = []
        for task in task_list:
            sub_tasks = get_subtasks(task)
            pairs += [(task, st) for st in sub_tasks]
    else:
        raise NotImplementedError(f"Task names {task_names} sub_tasks {sub_tasks} is not Implemented!")
    return pairs


def load_and_cache_multi_gen_data(args, pool, tokenizer, split_tag, only_src=False, is_sample=False):
    cp = os.path.join(args.run_output_dir, 'cache')
    os.makedirs(cp, exist_ok=True)
    cache_fn = os.path.join(cp, split_tag)
    if os.path.exists(cache_fn) and not is_sample:
        logger.info("Load cache data from %s", cache_fn)
        examples_data_dict = torch.load(cache_fn)
    else:
        examples_data_dict = {}

        task_names = args.task.split(',') if args.task is not None else None
        sub_tasks = args.sub_task.split(',') if args.sub_task != '' else None
        for task, sub_task in get_tasks_mtl(task_names, sub_tasks):
            if task == 'summarize':
                max_source_length = 256
                max_target_length = 128
            elif task == 'translate':
                max_source_length = 320
                max_target_length = 256
            elif task == 'refine':
                if sub_task == 'small':
                    max_source_length = 130
                    max_target_length = 120
                else:
                    max_source_length = 240
                    max_target_length = 240
            elif task == 'concode':
                max_source_length = 320
                max_target_length = 150
            elif task == 'defect':
                max_source_length = 512
                max_target_length = 3  # as do not need to add lang ids

            filename = get_filenames(args.data_dir, task, sub_task, split_tag)
            examples = read_examples(filename, args.data_num, task)
            if is_sample:
                examples = random.sample(examples, min(args.bleu_samples, len(examples)))
            # if split_tag == 'train':
            #     calc_stats(examples, tokenizer, is_tokenize=True)
            # else:
            #     calc_stats(examples)

            tuple_examples = [(example, idx, tokenizer, args, split_tag, task, sub_task, max_source_length, max_target_length) for idx, example in enumerate(examples)]
            if args.data_num == -1:
                features = pool.map(convert_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
            else:
                features = [convert_examples_to_features(x) for x in tuple_examples]
            all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
            if only_src:
                data = TensorDataset(all_source_ids)
            else:
                all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
                data = TensorDataset(all_source_ids, all_target_ids)
            examples_data_dict['{}_{}'.format(task, sub_task)] = (examples, data)

        if args.local_rank in [-1, 0] and not is_sample:
            torch.save(examples_data_dict, cache_fn)
            logger.info("Save data into %s", cache_fn)
    return examples_data_dict


def get_filenames(data_root, task, sub_task, split=''):
    if task == 'concode':
        data_dir = '{}/{}'.format(data_root, task)
        train_fn = '{}/train.json'.format(data_dir)
        dev_fn = '{}/dev.json'.format(data_dir)
        test_fn = '{}/test.json'.format(data_dir)
    elif task == 'summarize':
        data_dir = '{}/{}/{}'.format(data_root, task, sub_task)
        train_fn = '{}/train.jsonl'.format(data_dir)
        dev_fn = '{}/valid.jsonl'.format(data_dir)
        test_fn = '{}/test.jsonl'.format(data_dir)
    elif task == 'refine':
        data_dir = '{}/{}/{}'.format(data_root, task, sub_task)
        train_fn = '{}/train.buggy-fixed.buggy,{}/train.buggy-fixed.fixed'.format(data_dir, data_dir)
        dev_fn = '{}/valid.buggy-fixed.buggy,{}/valid.buggy-fixed.fixed'.format(data_dir, data_dir)
        test_fn = '{}/test.buggy-fixed.buggy,{}/test.buggy-fixed.fixed'.format(data_dir, data_dir)
    elif task == 'translate':
        data_dir = '{}/{}'.format(data_root, task)
        if sub_task == 'cs-java':
            train_fn = '{}/train.java-cs.txt.cs,{}/train.java-cs.txt.java'.format(data_dir, data_dir)
            dev_fn = '{}/valid.java-cs.txt.cs,{}/valid.java-cs.txt.java'.format(data_dir, data_dir)
            test_fn = '{}/test.java-cs.txt.cs,{}/test.java-cs.txt.java'.format(data_dir, data_dir)
        else:
            train_fn = '{}/train.java-cs.txt.java,{}/train.java-cs.txt.cs'.format(data_dir, data_dir)
            dev_fn = '{}/valid.java-cs.txt.java,{}/valid.java-cs.txt.cs'.format(data_dir, data_dir)
            test_fn = '{}/test.java-cs.txt.java,{}/test.java-cs.txt.cs'.format(data_dir, data_dir)
    elif task == 'clone':
        data_dir = '{}/{}'.format(data_root, task)
        train_fn = '{}/train.txt'.format(data_dir)
        dev_fn = '{}/valid.txt'.format(data_dir)
        test_fn = '{}/test.txt'.format(data_dir)
    elif task == 'defect':
        data_dir = '{}/{}'.format(data_root, task)
        train_fn = '{}/train.jsonl'.format(data_dir)
        dev_fn = '{}/valid.jsonl'.format(data_dir)
        test_fn = '{}/test.jsonl'.format(data_dir)
    if split == 'train':
        return train_fn
    elif split == 'dev':
        return dev_fn
    elif split == 'test':
        return test_fn
    else:
        return train_fn, dev_fn, test_fn


def read_examples(filename, data_num, task):
    read_example_dict = {
        'summarize': read_summarize_examples,
        'refine': read_refine_examples,
        'translate': read_translate_examples,
        'concode': read_concode_examples,
        'clone': read_clone_examples,
        'defect': read_defect_examples,
    }
    return read_example_dict[task](filename, data_num)


def calc_stats(examples, tokenizer=None, is_tokenize=False):
    avg_src_len = []
    avg_trg_len = []
    avg_src_len_tokenize = []
    avg_trg_len_tokenize = []
    for ex in examples:
        if is_tokenize:
            avg_src_len.append(len(ex.source.split()))
            avg_trg_len.append(len(str(ex.target).split()))
            avg_src_len_tokenize.append(len(tokenizer.tokenize(ex.source)))
            avg_trg_len_tokenize.append(len(tokenizer.tokenize(str(ex.target))))
        else:
            avg_src_len.append(len(ex.source.split()))
            avg_trg_len.append(len(str(ex.target).split()))
    if is_tokenize:
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))
        logger.info("[TOKENIZE] avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    np.mean(avg_src_len_tokenize), np.mean(avg_trg_len_tokenize), max(avg_src_len_tokenize),
                    max(avg_trg_len_tokenize))
    else:
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))


def get_elapse_time(t0):
    elapse_time = time.time() - t0
    if elapse_time > 3600:
        hour = int(elapse_time // 3600)
        minute = int((elapse_time % 3600) // 60)
        return "{}h{}m".format(hour, minute)
    else:
        minute = int((elapse_time % 3600) // 60)
        return "{}min".format(minute)


def setup_wandb_logger(args):
    wandb_writer = wandb.init(project=args.project_name, dir=args.output_dir, save_code=False, name=args.name, config=args)#, group=args.group)

    src_dir = Path(__file__).resolve().parent
    base_path = str(src_dir.parent)
    src_dir = str(src_dir)
    return wandb_writer, src_dir, base_path


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False





















def bleu_checkpoint_and_earlystop(args, results, best_metric, metric_key, model, filehandle, not_loss_dec_cnt, cur_epoch, mode='max',):
    early_stop = False
    dev_bleu_em = results['dev_bleu_em']
    dev_bleu = results['dev_bleu']
    dev_em = results['dev_em']

    if mode == 'max':
        better = results[metric_key] > best_metric
    elif mode == 'min':
        better = results[metric_key] < best_metric
    else:
        raise ValueError()

    if better:
        not_bleu_em_inc_cnt = 0
        logger.info("  [%d] Best bleu+em: %.2f (bleu: %.2f, em: %.2f)",
                    cur_epoch, dev_bleu_em, dev_bleu, dev_em)
        logger.info("  " + "*" * 20)
        best_metric = dev_bleu_em
        filehandle.write("[%d] Best bleu+em changed into %.2f (bleu: %.2f, em: %.2f)\n" % (
            cur_epoch, best_metric, dev_bleu, dev_em))
        # Save best checkpoint for best bleu
        output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if args.data_num == -1 or args.always_save_model:
            model_to_save = model.module if hasattr(model, 'module') else model
            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)
            logger.info("Save the best bleu model into %s", output_model_file)
    else:
        not_bleu_em_inc_cnt += 1
        logger.info("Bleu does not increase for %d epochs", not_bleu_em_inc_cnt)
        filehandle.write(
            "[%d] Best bleu+em (%.2f) does not drop changed for %d epochs, cur bleu+em: %.2f (bleu: %.2f, em: %.2f)\n" % (
                cur_epoch, best_metric, not_bleu_em_inc_cnt, dev_bleu_em, dev_bleu, dev_em))
        if all([x > args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]):
            stop_early_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                cur_epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt)
            logger.info(stop_early_str)
            filehandle.write(stop_early_str)
            early_stop = True
    return best_metric, early_stop

def ppl_checkpoint_and_earlystop(args, curr_metric, best_metric, model, filehandle, not_bleu_em_inc_cnt, cur_epoch,):
    early_stop = False

    if curr_metric < best_metric:
        not_loss_dec_cnt = 0
        logger.info("  Best ppl:%s", curr_metric)
        logger.info("  " + "*" * 20)
        filehandle.write("[%d] Best ppl changed into %.4f\n" % (cur_epoch, curr_metric))
        best_ppl = curr_metric

        # Save best checkpoint for best ppl
        output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if args.always_save_model:
            model_to_save = model.module if hasattr(model, 'module') else model
            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)
            logger.info("Save the best ppl model into %s", output_model_file)
    else:
        not_loss_dec_cnt += 1
        logger.info("Ppl does not decrease for %d epochs", not_loss_dec_cnt)
        if all([x > args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]):
            early_stop_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                cur_epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt)
            logger.info(early_stop_str)
            filehandle.write(early_stop_str)
            early_stop = True
    return best_ppl, early_stop

def load_and_cache_clone_data(args, filename, pool, tokenizer, split_tag, is_sample=False):
    cache_fn = '{}/{}.pt'.format(args.cache_path, split_tag + '_all' if args.data_num == -1 else '_%d' % args.data_num)
    examples = read_examples(filename, args.data_num, task)
    if is_sample:
        examples = random.sample(examples, int(len(examples) * 0.1))

    calc_stats(examples, tokenizer, is_tokenize=True)
    if os.path.exists(cache_fn):
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        if is_sample:
            logger.info("Sample 10 percent of data from %s", filename)
        elif args.data_num == -1:
            logger.info("Create cache data into %s", cache_fn)
        tuple_examples = [(example, idx, tokenizer, args) for idx, example in enumerate(examples)]
        features = pool.map(convert_clone_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        data = TensorDataset(all_source_ids, all_labels)

        if args.local_rank in [-1, 0] and args.data_num == -1:
            torch.save(data, cache_fn)
    return examples, data

def load_and_cache_defect_data(args, filename, pool, tokenizer, split_tag, is_sample=False):
    cache_fn = os.path.join(args.cache_path, split_tag)
    examples = read_examples(filename, args.data_num, task)
    if is_sample:
        examples = random.sample(examples, int(len(examples) * 0.1))

    calc_stats(examples, tokenizer, is_tokenize=True)
    if os.path.exists(cache_fn):
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        if is_sample:
            logger.info("Sample 10 percent of data from %s", filename)
        elif args.data_num == -1:
            logger.info("Create cache data into %s", cache_fn)
        tuple_examples = [(example, idx, tokenizer, args) for idx, example in enumerate(examples)]
        features = pool.map(convert_defect_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
        # features = [convert_clone_examples_to_features(x) for x in tuple_examples]
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        data = TensorDataset(all_source_ids, all_labels)

        if args.local_rank in [-1, 0] and args.data_num == -1:
            torch.save(data, cache_fn)
    return examples, data
