# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

import os, sys
import logging
import argparse
import numpy as np
from tqdm import tqdm
import time, subprocess
from collections import defaultdict as ddict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset
from models.models import build_or_load_gen_model
from evaluator import smooth_bleu
from evaluator.CodeBLEU import calc_code_bleu
from evaluator.bleu import _bleu
from utils.utils import *
from utils.metrics import *
from utils.configs import *
from utils.replay import Buffer
from models.T5prompt import PromptTuneT5
from dataloaders.generation_loader import CodeXGlueDataModule, BigQueryDataModule

BIGQUERY = ['database', 'gui', 'networking', 'science', 'web']

def val_ppl_epoch(args, val_dataloader, model, tokenizer, eval_task=None):

    model.eval()
    val_loss, batch_num = 0, 0
    for batch in tqdm(val_dataloader, total=len(val_dataloader), desc="Eval ppl"):
        batch = tuple(t.to(args.device) for t in batch)
        source_ids, target_ids = batch
        source_mask = source_ids.ne(tokenizer.pad_token_id).type(torch.float)
        target_mask = target_ids.ne(tokenizer.pad_token_id).type(torch.float)

        kwargs = {}
        with torch.no_grad():
            if args.model_type == 'roberta':
                loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                   target_ids=target_ids, target_mask=target_mask)
            else:
                if args.num_prompts_per_task > 0 or args.prompt_method == "pool":
                    kwargs['task_name'] = eval_task
                    kwargs['phase'] = 'eval'
                outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                labels=target_ids, decoder_attention_mask=target_mask, **kwargs)
                loss = outputs[0].loss.mean() if args.num_prompts_per_task > 0 or args.prompt_method == "pool" else outputs.loss.mean()

                if "pool" in args.prompt_method:
                    prompt_loss = outputs[1]
                    loss = loss + prompt_loss.mean(dim=0)

        val_loss += loss.item()
        batch_num += 1
    val_loss = val_loss / batch_num
    val_ppl = round(np.exp(val_loss), 5)
    return val_ppl

def val_bleu_epoch(args, val_dataloader, val_examples, model, tokenizer, max_target_length, curr_task, eval_task, fname, logger):
    model.eval()
    pred_ids = []
    bleu, codebleu = 0.0, 0.0

    kwargs = { 'use_cache':True, 'num_beams': args.beam_size, 'early_stopping': (args.task == 'summarize'), 'max_length': max_target_length }
    for batch in tqdm(val_dataloader, total=len(val_dataloader), desc=f"Train Task: {curr_task} Eval Task {eval_task}: Eval bleu"):
        source_ids = batch[0].to(args.device)
        if args.io_queries:
            target_ids = batch[1].to(args.device)
        else:
            target_ids = None
        source_mask = source_ids.ne(tokenizer.pad_token_id).type(torch.float)
        with torch.no_grad():
            if args.model_type == 'roberta':
                preds = model(source_ids=source_ids, source_mask=source_mask)
                top_preds = [pred[0].cpu().numpy() for pred in preds]
            else:
                if hasattr(model, 'module'):
                    model_to_generate = model.module
                else:
                    model_to_generate = model
                if args.num_prompts_per_task > 0 or args.prompt_method == "pool":
                    preds = model_to_generate.generate(source_ids, source_mask, target_ids, eval_task, kwargs)
                else:
                    preds = model_to_generate.generate(source_ids, attention_mask=source_mask, **kwargs)
                top_preds = list(preds.cpu().numpy())
            pred_ids.extend(top_preds)

    pred_nls = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]

    args.prediction_dir = os.path.join(args.run_output_dir, 'prediction')
    os.makedirs(args.prediction_dir, exist_ok=True)
    output_fn = os.path.join(args.prediction_dir, "test_{}.output".format(fname))
    gold_fn = os.path.join(args.prediction_dir, "test_{}.gold".format(fname))
    src_fn = os.path.join(args.prediction_dir, "test_{}.src".format(fname))

    if 'defect' in eval_task:
        target_dict = {0: 'false', 1: 'true'}
        golds = [target_dict[ex.target] for ex in val_examples]
        val_acc = np.mean([int(p == g) for p, g in zip(pred_nls, golds)])
        results = {'em': val_acc * 100, 'bleu': 0, 'codebleu': 0}

        with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
            for pred_nl, gold in zip(pred_nls, val_examples):
                f.write(pred_nl.strip() + '\n')
                f1.write(target_dict[gold.target] + '\n')
                f2.write(gold.source.strip() + '\n')
            logger.info("Save the predictions into %s", output_fn)
    else:
        dev_accs, predictions = [], []
        with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
            for pred_nl, gold in zip(pred_nls, val_examples):
                dev_accs.append(pred_nl.strip() == gold.target.strip())
                if 'summarize' in eval_task:
                    # for smooth-bleu4 evaluation
                    predictions.append(str(gold.idx) + '\t' + pred_nl)
                    f.write(str(gold.idx) + '\t' + pred_nl.strip() + '\n')
                    f1.write(str(gold.idx) + '\t' + gold.target.strip() + '\n')
                    f2.write(str(gold.idx) + '\t' + gold.source.strip() + '\n')
                else:
                    f.write(pred_nl.strip() + '\n')
                    f1.write(gold.target.strip() + '\n')
                    f2.write(gold.source.strip() + '\n')

        if 'summarize' in eval_task:
            (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions, gold_fn)
            bleu = round(smooth_bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
        else:
            bleu = round(_bleu(gold_fn, output_fn), 2)
            if any([tag in eval_task for tag in ['concode', 'translate', 'refine']]):
                codebleu = calc_code_bleu.get_codebleu(gold_fn, output_fn, getlang(eval_task))

        results = {'em': np.mean(dev_accs) * 100, 'bleu': bleu, "codebleu": 0}
        if 'concode' in eval_task:
            results['codebleu'] = codebleu * 100

    if 'summarize' in eval_task:
        results['bleu_em'] = results['bleu']
    elif 'defect' in eval_task:
        results['bleu_em'] = results['em']
    else:
        results['bleu_em'] = results['bleu'] + results['em']

    logger.info(f"Train Task: {curr_task} Eval Task: {eval_task} \t " + "  ".join(f'{k}={v:.4f}' for k,v in sorted(results.items())))
    return results

def val_bleu_epoch_bigquery(args, val_dataloader, model, tokenizer, split, max_target_length, curr_task, eval_task, fname, logger):
    model.eval()
    pred_ids = []
    val_examples = []
    bleu, codebleu = 0.0, 0.0

    kwargs = { 'use_cache':True, 'num_beams': args.beam_size, 'early_stopping': (args.task == 'summarize'), 'max_length': max_target_length }
    for batch in tqdm(val_dataloader, total=len(val_dataloader), desc=f"Train Task: {curr_task} Eval Task {eval_task}: Eval bleu"):
        source_ids = batch[0].to(args.device)
        val_examples += batch[2]
        if args.io_queries:
            target_ids = batch[1].to(args.device)
        else:
            target_ids = None
        source_mask = source_ids.ne(tokenizer.pad_token_id).type(torch.float)
        with torch.no_grad():
            if args.model_type == 'roberta':
                preds = model(source_ids=source_ids, source_mask=source_mask)
                top_preds = [pred[0].cpu().numpy() for pred in preds]
            else:
                if hasattr(model, 'module'):
                    model_to_generate = model.module
                else:
                    model_to_generate = model
                if args.num_prompts_per_task > 0 or args.prompt_method == "pool":
                    preds = model_to_generate.generate(source_ids, source_mask, target_ids, eval_task, kwargs)
                else:
                    preds = model_to_generate.generate(source_ids, attention_mask=source_mask, **kwargs)
                top_preds = list(preds.cpu().numpy())
            pred_ids.extend(top_preds)

    pred_nls = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]
    if split == "test":
        pred_nls = [p.split("\n")[0] for p in pred_nls]


    args.prediction_dir = os.path.join(args.run_output_dir, 'prediction')
    os.makedirs(args.prediction_dir, exist_ok=True)
    output_fn = os.path.join(args.prediction_dir, "test_{}.output".format(fname))
    gold_fn = os.path.join(args.prediction_dir, "test_{}.gold".format(fname))
    src_fn = os.path.join(args.prediction_dir, "test_{}.src".format(fname))

    dev_accs = []
    with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
        for pred_nl, gold in zip(pred_nls, val_examples):
            dev_accs.append(pred_nl.strip() == gold.target.strip())
            f.write(pred_nl.strip().replace("\n", "\\n").replace("\t", "\\t") + '\n')
            f1.write(gold.target.strip().replace("\n", "\\n").replace("\t", "\\t") + '\n')
            f2.write(gold.source.strip().replace("\n", "\\n").replace("\t", "\\t") + '\n')

    bleu = round(_bleu(gold_fn, output_fn), 2)
    # codebleu = calc_code_bleu.get_codebleu(gold_fn, output_fn, "python")
    results = {'em': np.mean(dev_accs) * 100, 'bleu': bleu, "codebleu": codebleu * 100}

    results['bleu_em'] = results['bleu']

    logger.info(f"Train Task: {curr_task} Eval Task: {eval_task} \t " + "  ".join(f'{k}={v:.4f}' for k,v in sorted(results.items())))
    return results

def train_epoch(args, model, dataloader, tokenizer, optimizer, scheduler, curr_epoch, curr_task, buffer=None, writer=None):
    bar = tqdm(dataloader, total=len(dataloader), desc="Training")
    nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
    model.train()
    kwargs = {}
    for step, batch in enumerate(bar):
        batch = tuple(t.to(args.device) for t in batch)
        source_ids, target_ids = batch
        source_mask = source_ids.ne(tokenizer.pad_token_id).type(torch.float)
        target_mask = target_ids.ne(tokenizer.pad_token_id).type(torch.float)

        if args.model_type == 'roberta':
            loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                            target_ids=target_ids, target_mask=target_mask)
            loss = loss.mean()
        else:
            if args.num_prompts_per_task > 0 or args.prompt_method == "pool":
                kwargs['task_name'] = curr_task
                kwargs['phase'] = 'train'
            outputs = model(input_ids=source_ids, attention_mask=source_mask,
                            labels=target_ids, decoder_attention_mask=target_mask, **kwargs)

            loss = outputs[0].loss.mean() if args.num_prompts_per_task > 0 or args.prompt_method == "pool" else outputs.loss.mean()

            if "pool" in args.prompt_method:
                prompt_loss = outputs[1]
                loss = loss + prompt_loss.mean(dim=0)

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        tr_loss += loss.item()

        nb_tr_examples += source_ids.size(0)
        nb_tr_steps += 1
        loss.backward()

        if args.replay == 'reservoir' and args.buffer_size > 0:
            buffer.add_data(examples=source_ids, labels=target_ids)

        if nb_tr_steps % args.gradient_accumulation_steps == 0:
            # Update parameters
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)

            if not args.debug:
                writer.log({'train_loss': train_loss})
                writer.log({'raw_loss': loss})
                writer.log({f'{curr_task}/train_loss': train_loss})
                writer.log({f'lr': optimizer.param_groups[0]['lr']})
            bar.set_description(f"[Task: {curr_task} Epoch:{curr_epoch}] Train loss {train_loss:.3f}")

def epoch_end_replay_buffer(args, buffer, batch_size, model, tokenizer, optimizer,):
    batch_size = args.n_gpu * batch_size
    learned_tasks = list(buffer.task2idx.keys())
    random.shuffle(learned_tasks)
    for task in learned_tasks:
        indices = buffer.task2idx[task]
        random.shuffle(indices)
        for i in range(len(indices)//batch_size + 1):
            # cycle through list to avoid weird batch sizes and corresponding error in distributed training.
            batch_indices = (indices*2)[i * batch_size : (i+1) * batch_size]
            # print(batch_indices)
            if len(batch_indices) == 0: continue
            source_ids, target_ids, task_ids = buffer.get_data_by_index(indexes=batch_indices)
            assert len(torch.unique(task_ids)) == 1, f"Multiple ({len(torch.unique(task_ids))}) buffer tasks in the same batch"
            source_mask = source_ids.ne(tokenizer.pad_token_id).type(torch.float)
            target_mask = target_ids.ne(tokenizer.pad_token_id).type(torch.float)
            if args.model_type == 'roberta':
                buff_loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                target_ids=target_ids, target_mask=target_mask)
            else:
                kwargs = {}
                if args.num_prompts_per_task > 0 or args.prompt_method == "pool":
                    kwargs['task_name'] = task
                    kwargs['phase'] = 'train'
                outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                labels=target_ids, decoder_attention_mask=target_mask, **kwargs)

                loss = outputs[0].loss.mean() if args.num_prompts_per_task > 0 or args.prompt_method == "pool" else outputs.loss.mean()

                if "pool" in args.prompt_method:
                    prompt_loss = outputs[1]
                    loss = loss + prompt_loss.mean(dim=0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def task_save_buffer_samples(args, buffer, dataset, task_idx):
    all_tasks = args.stream.split(',')
    buffer_src_len, buffer_tgt_len = max(args.max_source_length), max(args.max_target_length)
    num_store_examples = buffer.buffer_portion_size
    store_indices = torch.randint(0, len(dataset), (num_store_examples,))
    if all_tasks[task_idx] in ["gui", "web", "networking", "science", "database"]:
        selected_data = [dataset[idx] for idx in store_indices]
        source_ids, target_ids = dataset.collate_fn(selected_data)
    else:
        source_ids, target_ids = dataset[store_indices]
    source_ids = F.pad(source_ids, pad=(0, buffer_src_len - source_ids.shape[1], 0, 0))
    target_ids = F.pad(target_ids, pad=(0, buffer_tgt_len - target_ids.shape[1], 0, 0))
    task_labels = torch.ones(num_store_examples, dtype=int) * task_idx
    buffer.add_data(examples=source_ids, labels=target_ids, task_labels=task_labels)
    return buffer

def main():
    t0 = time.time()
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    args.org_pool_size = args.pool_size

    file_handler = logging.FileHandler(filename=f"{args.log_dir}/all.log")
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[file_handler, stdout_handler]
                        )
    logger = logging.getLogger(__name__)
    logger.info(args)
    logger.info(args.run_output_dir.upper())

    set_dist(args)
    seed_everything(args.seed)
    if args.local_rank in [-1, 0] and not args.debug:
        wandb_writer, _, _ = setup_wandb_logger(args)
        wandb_writer.log({"command": subprocess.list2cmdline(['python'] + sys.argv)})
    else:
        wandb_writer = None
        logger.info(f"{subprocess.list2cmdline(['python'] + sys.argv)}")

    config, model, tokenizer = build_or_load_gen_model(args)

    if all([t in BIGQUERY for t in args.stream.split(',')]):
        datamodule = BigQueryDataModule(args, tokenizer)
    else:
        datamodule = CodeXGlueDataModule(args, tokenizer)
    all_tasks = datamodule.all_tasks
    task_specific_params = ['num_train_epochs', 'learning_rate', 'patience', 'max_source_length',
                            'max_target_length', 'train_batch_size', 'eval_batch_size']
    args = get_task_arglist(args, task_specific_params, datamodule)
    datamodule.setup(stage='fit')

    if args.num_prompts_per_task > 0 or args.prompt_method == "pool":
        logger.info(f"***** Prompt Learning Rate: {args.prompt_lr} *****")
        model = PromptTuneT5(args, model, tokenizer, datamodule)

    model.to(args.device)
    if 'pool' in args.prompt_method:
        model.initialize_prompt_pool()
        model.to(args.device)
    model_to_save = model.module if hasattr(model, 'module') else model
    os.makedirs(os.path.join(args.run_output_dir, 'checkpoints'), exist_ok=True)
    torch.save(model_to_save.state_dict(), os.path.join(args.run_output_dir, 'checkpoints', 'first.bin'))


    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    fa_result = open(os.path.join(args.run_output_dir, 'results.csv'), 'a+', 1)
    fa = open(os.path.join(args.log_dir, 'summary.log'), 'a+', 1)
    fa_cl = open(os.path.join(args.log_dir, 'cl_eval.log'), 'a+', 1)
    fa_dict = {}
    for tt in all_tasks:
        fa_dict[tt] = open(os.path.join(args.log_dir, f'{tt}.log'), 'a+', 1)

    if not args.no_train:
        # datamodule.setup(stage='fit')
        if args.replay != '' and args.buffer_size > 0:
            buffer = Buffer(args.buffer_size, args.device, len(all_tasks), mode=args.replay)
            args.replay_epoch_end = True if args.replay != 'reservoir' else False
        else:
            buffer = None

        train_dataloaders = datamodule.train_dataloader()
        val_dataloader = datamodule.val_dataloader()
        # if not args.no_eval or not args.no_eval_all:
        #     val_dataloader = datamodule.val_dataloader()

        # Storing varaible across tasks.
        step, global_step = 0, 0
        best_bleu_em = dict([(k, -1) for k in all_tasks])
        best_ppl = dict([(k, 1e6) for k in all_tasks])
        early_stop_ppl = dict([(k, False) for k in all_tasks])
        early_stop_bleu = dict([(k, False) for k in all_tasks])
        not_dec_cnt = ddict(dict)
        cl_bleu = np.ones((len(all_tasks), len(all_tasks))) * 0
        cl_codebleu = np.ones((len(all_tasks), len(all_tasks))) * 0
        cl_ppl = np.ones((len(all_tasks), len(all_tasks))) * 1e6
        cl_em = np.ones((len(all_tasks), len(all_tasks))) * 0

        # Start training
        logger.info(f"***** Running training on {len(all_tasks)} tasks *****")
        logger.info(f"Tasks: {all_tasks}\t")
        logger.info(f"Epochs: {args.num_train_epochs}\t")
        logger.info(f"LR: {args.learning_rate}\t")
        logger.info(f"Patience: {args.patience}\t")
        logger.info(f"Train Batch size: {args.train_batch_size}\t")
        logger.info(f"Eval Batch size: {args.eval_batch_size}\t")
        logger.info(f"Max_src_lens: {args.max_source_length}\t")
        logger.info(f"Max_tgt_lens: {args.max_target_length}\t")
        logger.info("  " + "*" * 20)

        for curr_idx, curr_task in enumerate(all_tasks):
            not_dec_cnt[curr_task]['val_ppl'], not_dec_cnt[curr_task]['bleu_em'] = 0, 0 if not args.no_eval_bleu else 1e6

            # Setup model, dataloader, and optimizer.
            task_lr = args.learning_rate[curr_idx]
            task_patience = args.patience[curr_idx]
            task_epochs = args.num_train_epochs[curr_idx]
            train_loader = train_dataloaders[curr_task]
            task_base_train_bs = args.train_batch_size[curr_idx]
            task_max_source_length = args.max_source_length[curr_idx]
            task_max_target_length = args.max_target_length[curr_idx]
            assert args.train_batch_size[curr_idx] * args.n_gpu == train_loader.batch_size, f"Train batch size doesn't match for task {curr_task}"
            logger.info(f"Training on Task {curr_task} for {task_epochs} Epochs")

            effective_global_batch_size = (args.n_gpu * task_base_train_bs * args.gradient_accumulation_steps)
            factor = effective_global_batch_size / task_base_train_bs
            task_lr = task_lr * np.sqrt(factor) if factor > 1 else task_lr
            warmup_steps = args.warmup_steps // args.n_gpu
            logger.info(f"Inferred LR as {task_lr:.6f} based on sqrt scaling rule (BS scale {factor:.2f}x).")
            logger.info(f"Inferred warmup steps as {args.warmup_steps} based on sqrt scaling rule (BS scale {factor:.2f}x).")
            logger.info(f"Train Batch Size: {task_base_train_bs}\t BLEU eval Batch Size: {args.eval_batch_size[curr_idx]}")
            logger.info(f"Max Src Len: {task_max_source_length}\t Max Tgt Len: {task_max_target_length}")
            logger.info(f"Train: {len(train_loader) * train_loader.batch_size}\tVal: {len(val_dataloader[curr_task])*val_dataloader[curr_task].batch_size}")
            optimizer, scheduler = get_optimizer(args, model, train_loader, lr=task_lr, warmup_steps=warmup_steps, epochs=task_epochs)

            for curr_epoch in range(0, int(task_epochs)):
                train_epoch(args, model, train_loader, tokenizer, optimizer, scheduler, curr_epoch, curr_task, buffer, wandb_writer)
                global_step += len(train_loader) // args.gradient_accumulation_steps

                torch.cuda.empty_cache()

                if args.replay != 'reservoir' and args.buffer_size > 0 and curr_epoch == 0:
                    logger.info('Logging Samples at Task end.')
                    buffer = task_save_buffer_samples(args, buffer, datamodule.train_data[curr_task], curr_idx)

                if args.replay_epoch_end and len(buffer) > 0:
                    logger.info('Epoch End Replaying Buffer')
                    epoch_end_replay_buffer(args, buffer, args.buffer_bs, model, tokenizer, optimizer,)

                # save last checkpoint
                if args.save_last_checkpoints:
                    last_output_dir = os.path.join(args.run_output_dir, 'checkpoints')
                    os.makedirs(last_output_dir, exist_ok=True)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(last_output_dir, "last.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("Save the last model into %s", output_model_file)

                if not args.no_eval:
                    # Eval model with dev dataset on current task for early stopping and checkpointing.
                    results = {}
                    assert args.train_batch_size[curr_idx] * args.n_gpu == val_dataloader[curr_task].batch_size, f"Eval PPL Batch size doesn't match for task {curr_task}"
                    val_ppl = val_ppl_epoch(args, val_dataloader[curr_task], model, tokenizer, eval_task=curr_task)
                    results_ppl = {'epoch': curr_epoch, 'global_step': global_step, 'val_ppl': val_ppl}
                    logger.info(f"Train Task: {curr_task} Eval Task: {curr_task}: "+ " ".join(f'{k}={v}' for k,v in sorted(results_ppl.items())))
                    if not args.debug: wandb_writer.log({f"{curr_task}/dev_ppl": val_ppl})
                    best_ppl[curr_task], early_stop_ppl[curr_task], not_dec_cnt[curr_task] = checkpoint_and_earlystop(args, results_ppl, best_ppl[curr_task],\
                        'val_ppl', model, fa_dict[curr_task], not_dec_cnt[curr_task], curr_epoch, curr_task, mode='min', patience=task_patience)

                    torch.cuda.empty_cache()

                    if not args.no_eval_bleu:
                        savename = f'{curr_task}_{curr_task}_e{curr_epoch:.0f}'
                        if curr_task in ["gui", "web", "networking", "science", "database"]:
                            bleu_dataloader = datamodule.get_bleu_dataloader(curr_task)
                            assert args.eval_batch_size[curr_idx] == bleu_dataloader.batch_size, f"Eval Bleu Batch size doesn't match for task {curr_task}"
                            results_bleu = val_bleu_epoch_bigquery(args, bleu_dataloader, model, tokenizer, "val", task_max_target_length, curr_task, curr_task, savename, logger)
                        else:
                            val_bleu_examples, val_bleu_data, bleu_dataloader = datamodule.get_bleu_dataloader(curr_task)
                            assert args.eval_batch_size[curr_idx] == bleu_dataloader.batch_size, f"Eval Bleu Batch size doesn't match for task {curr_task}"
                            results_bleu = val_bleu_epoch(args, bleu_dataloader, val_bleu_examples, model, tokenizer, task_max_target_length, curr_task, curr_task, savename, logger)
                        dev_bleu, dev_em, dev_bleu_em = results_bleu['bleu'], results_bleu['em'], results_bleu['bleu_em']

                        if not args.debug:
                            log_dict = {f"{curr_task}/dev_{k}": v for k,v in results_bleu.items()}
                            wandb_writer.log(log_dict)

                        best_bleu_em[curr_task], early_stop_bleu[curr_task], not_dec_cnt[curr_task] = checkpoint_and_earlystop(args, results_bleu, best_bleu_em[curr_task],\
                            'bleu_em', model, fa_dict[curr_task], not_dec_cnt[curr_task], curr_epoch, curr_task, mode='max', patience=task_patience)

                if not args.no_eval_bleu:
                    if early_stop_ppl[curr_task] and early_stop_bleu[curr_task]:
                        break
                else:
                    if early_stop_ppl[curr_task]:
                        break

                # torch.cuda.empty_cache()
                logger.info("  " + "*" * 20)

            if not args.no_eval_all:
                # del model
                # torch.cuda.empty_cache()

                logger.info(f"***** Eval results after training on Task {curr_task} *****".upper())
                file = os.path.join(args.run_output_dir, f'checkpoints/best-bleu.bin')
                logger.info(f"Reload best model from {file}")
                loaded_state_dict = torch.load(file)
                if isinstance(model, nn.DataParallel):
                    for buf in model.module._buffers.keys():
                        model.module._buffers[buf] = torch.empty_like(loaded_state_dict[buf])
                    model.module.load_state_dict(loaded_state_dict)
                else:
                    for buf in model._buffers.keys():
                        model._buffers[buf] = torch.empty_like(loaded_state_dict[buf])
                    model.load_state_dict(loaded_state_dict)

                fa_result.write(f"\ncurr_task,eval_task,PPL,BLEU,CodeBLEU,EM\n")
                for eval_idx, eval_task in enumerate(all_tasks):
                    if eval_idx > curr_idx and not args.full_matrix_eval: break
                    val_ppl = val_ppl_epoch(args, val_dataloader[eval_task], model, tokenizer, eval_task=eval_task)
                    # # Bleu on sampled data faster but not comparable across tasks.
                    # val_bleu_examples, val_bleu_data, bleu_dataloader = datamodule.get_bleu_dataloader(eval_task)
                    savename = f'{curr_task}_{eval_task}_end'
                    if curr_task in ["gui", "web", "networking", "science", "database"]:
                        bleu_dataloader = datamodule.get_bleu_dataloader(curr_task, all_bleu=True)
                        assert args.train_batch_size[eval_idx] * args.n_gpu == val_dataloader[eval_task].batch_size, f"Eval PPL Batch size doesn't match for task {eval_task}"
                        assert args.eval_batch_size[eval_idx] == bleu_dataloader.batch_size, f"Eval Bleu Batch size doesn't match for task {eval_task}"
                        results = val_bleu_epoch_bigquery(args, bleu_dataloader, model, tokenizer, "val", args.max_target_length[eval_idx], curr_task, eval_task, savename, logger)
                    else:
                        val_bleu_examples, val_bleu_data, bleu_dataloader = datamodule.get_bleu_dataloader(eval_task, all_bleu=True)
                        assert args.train_batch_size[eval_idx] * args.n_gpu == val_dataloader[eval_task].batch_size, f"Eval PPL Batch size doesn't match for task {eval_task}"
                        assert args.eval_batch_size[eval_idx] == bleu_dataloader.batch_size, f"Eval Bleu Batch size doesn't match for task {eval_task}"
                        results = val_bleu_epoch(args, bleu_dataloader, val_bleu_examples, model, tokenizer, args.max_target_length[eval_idx], curr_task, eval_task, savename, logger)

                    cl_bleu[curr_idx, eval_idx] = results['bleu']
                    cl_codebleu[curr_idx, eval_idx] = results['codebleu']
                    cl_em[curr_idx, eval_idx] = results['em']
                    cl_ppl[curr_idx, eval_idx] = val_ppl

                    fa_result.write(f"{curr_task},{eval_task},{val_ppl},{results['bleu']},{results['codebleu']},{results['em']}\n")
                    msg = f"TT:{curr_task} ET:{eval_task}\tPPL:{val_ppl:.4f} " + " ".join(f"{k}:{v:.4f}" for k,v in results.items()) + "\n"
                    fa_cl.write(msg)
                    if not args.debug:
                        wandb_writer.log({
                            f'{eval_task}/cl_ppl': val_ppl,
                            f'{eval_task}/cl_bleu': results['bleu'],
                            f'{eval_task}/cl_codebleu': results['codebleu'],
                            f'{eval_task}/cl_em': results['em'],
                            }, step=curr_idx)
                    pass
                fa_result.write(f"\n")

            torch.cuda.empty_cache()
            logger.info(f"Finish training Task {curr_task}: {get_elapse_time(t0)}".upper())
            logger.info("  " + "*" * 30)
            logger.info("\n\n\n\n")

            if 'pool' in args.prompt_method and (args.pool_freq_norm or args.pool_freq):
                if isinstance(model, nn.DataParallel):
                    model.module._buffers['curr_count'] += model.module._buffers[f'{curr_task}_count']
                else:
                    model._buffers['curr_count'] += model._buffers[f'{curr_task}_count']

        if 'pool' in args.prompt_method and args.compute_avg_sim:
            task_rep_similarity = get_reps_cosine_similarity(model)
            logger.info(f"Task representation similarity: {task_rep_similarity.tolist()}")

        if 'pool' in args.prompt_method and (args.pool_freq_norm or args.pool_freq):
            ind_model = model.module if isinstance(model, nn.DataParallel) else model
            for k,v in ind_model._buffers.items():
                if 'count' in k:
                    logger.info(f"Pool Freq {k}: {v.type(torch.int).tolist()}")

        logger.info(f"Finish training All Tasks: {get_elapse_time(t0)}".upper())

        if len(all_tasks) > 1:
            forget_bleu = get_forgetting_metric(cl_bleu, bwt=False)
            forget_codebleu = get_forgetting_metric(cl_codebleu, bwt=False)
            forget_em = get_forgetting_metric(cl_em, bwt=False)
            forget_ppl = -get_forgetting_metric(-cl_ppl, bwt=False)
            log_dict = {'forget_bleu':forget_bleu, 'forget_em':forget_em, 'forget_ppl':forget_ppl}
            if not args.debug: wandb_writer.log(log_dict)

            fa_result.write(f"\n\n\nForgetting Metrics\n")
            fa_result.write(f"Avg. Forgetting\tBleu: {forget_bleu.mean():.4f} " + f"EM: {forget_em.mean():.4f} " + f"PPL: {forget_ppl.mean():.4f}\n")
            fa_result.write("metric," + ','.join(f"{tt}" for tt in all_tasks[:-1]) + "\n")
            fa_result.write("BLEU," + ','.join(f"{tt:.4f}" for tt in forget_bleu) + "\n")
            fa_result.write("CodeBLEU," + ','.join(f"{tt:.4f}" for tt in forget_codebleu) + "\n")
            fa_result.write("EM," + ','.join(f"{tt:.4f}" for tt in forget_em) + "\n")
            fa_result.write("PPL," + ','.join(f"{tt:.4f}" for tt in forget_ppl) + "\n")

            msg_forget = f"Forgetting for Bleu: " + ", ".join(f'{fo:.4f}' for fo in forget_bleu) + '\n' + \
                        f"Forgetting for CodeBleu: " + ", ".join(f'{fo:.4f}' for fo in forget_codebleu) + '\n' + \
                        f"Forgetting for Eaxct Match: " + ", ".join(f'{fo:.4f}' for fo in forget_em) + '\n' + \
                        f"Forgetting for PPL: " + ", ".join(f'{fo:.4f}' for fo in forget_ppl) + '\n' + \
                        f"Avg. Forgetting\tBleu: {forget_bleu.mean():.4f} CodeBleu: {forget_codebleu.mean():.4f} " + f"Exact Match: {forget_em.mean():.4f} " + f"PPL: {forget_ppl.mean():.4f}"
            fa_cl.write(msg_forget)
            logger.info(msg_forget)
            logger.info("  " + "*" * 40)

    if not args.no_test:
        logger.info("\n\n\n\n")
        logger.info("***** Testing *****".upper())
        logger.info(f"  Batch size = {args.eval_batch_size}")
        datamodule.setup(stage='test')
        if curr_task in ["gui", "web", "networking", "science", "database"]:
            test_dataloader = datamodule.test_dataloader()
        else:
            test_examples, test_data, test_dataloader = datamodule.test_dataloader()

        test_bleu, test_em, test_codebleu = np.empty(len(all_tasks)), np.empty(len(all_tasks)), np.empty(len(all_tasks))

        for criteria in ['best-bleu']:#, 'best-ppl', 'last']:
            _, model, _ = build_or_load_gen_model(args)
            if args.num_prompts_per_task > 0 or args.prompt_method == "pool":
                model = PromptTuneT5(args, model, tokenizer, datamodule)
            model.to(args.device)
            if "pool" in args.prompt_method:
                model.initialize_prompt_pool(load=False)
                model.to(args.device)

            if not args.zeroshot:
                file = os.path.join(args.run_output_dir, f'checkpoints/{criteria}.bin')
                loaded_state_dict = torch.load(file)
                for buf in model._buffers.keys():
                    model._buffers[buf] = torch.empty_like(loaded_state_dict[buf])
                model.load_state_dict(loaded_state_dict)
            else:
                logger.info("  " + "*" * 40)
                logger.info(f"performing zeroshot evaluation".upper())
                logger.info("  " + "*" * 40)
                file = 'huggingface CodeT5'
                curr_task = 'zeroshot'
                criteria = 'zeroshot CodeT5'
            model.to(args.device)

            reload_msg = f"\n\nTesting on model from {file}\n\n".upper()
            fa_result.write(reload_msg)
            logger.info(reload_msg)
            fa_cl.write(reload_msg)
            fa.write(reload_msg)

            fa_result.write(f"model,task,bleu,em,codebleu\n")
            for test_idx, test_task in enumerate(all_tasks):
                fa_dict[test_task].write(reload_msg)

                savename = f'{curr_task}_{eval_task}_end'
                if curr_task in ["gui", "web", "networking", "science", "database"]:
                    assert args.eval_batch_size[test_idx] == test_dataloader[test_task].batch_size, f"Test dataloader Batch size doesn't match for task {test_task}"
                    result = val_bleu_epoch_bigquery(args, test_dataloader[test_task], model, tokenizer, "test", args.max_target_length[test_idx], curr_task, test_task, criteria, logger)
                else:
                    assert args.eval_batch_size[test_idx] == test_dataloader[test_task].batch_size, f"Test dataloader Batch size doesn't match for task {test_task}"
                    result = val_bleu_epoch(args, test_dataloader[test_task], test_examples[test_task], model, tokenizer, args.max_target_length[test_idx], curr_task, test_task, criteria, logger)

                # result = val_bleu_epoch(args, test_dataloader[test_task], test_examples[test_task], model, tokenizer, args.max_target_length[test_idx], curr_task, test_task, criteria, logger)
                test_bleu[test_idx], test_em[test_idx] = result['bleu'], result['em']
                test_codebleu[test_idx] = result['codebleu'] if 'codebleu' in result else 0
                result_str = f"[Model: {criteria} Task: {test_task}] " + " ".join(f"{k}:{v:.4f}" for k,v in result.items()) + "\n"
                fa_result.write(f"{criteria},{test_task},{result['bleu']},{result['em']},{test_codebleu[test_idx]}\n")

                if not args.debug:
                    for k,v in result.items():
                        wandb_writer.log({f'test-{criteria}-{test_task}-{k}': v})

                logger.info(result_str)
                fa.write(result_str)
                fa_dict[test_task].write(result_str)
                fa_cl.write(result_str)

    logger.info("Finished and took {}".format(get_elapse_time(t0)))
    fa.write("Finished and took {}".format(get_elapse_time(t0)))
    fa.close()
    fa_result.close()
    fa_cl.close()
    for cur_task in all_tasks:
        fa_dict[cur_task].close()


if __name__ == "__main__":
    main()

