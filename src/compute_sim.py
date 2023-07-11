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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from models.models import build_or_load_gen_model
from evaluator import smooth_bleu
from evaluator.CodeBLEU import calc_code_bleu
from evaluator.bleu import _bleu
from utils.utils import *
from utils.metrics import *
from utils.configs import *
from utils.replay import Buffer
from models.T5prompt import PromptTuneT5
from dataloaders.generation_loader import CodeXGlueDataModule


def get_task_queries(args, dataloader, model, tokenizer, eval_task=None):
    model.eval()
    task_sims = []
    for batch in tqdm(dataloader, total=len(dataloader), desc="Eval ppl"):
        batch = tuple(t.to(args.device) for t in batch)
        source_ids, target_ids = batch
        targets = target_ids if args.io_queries else None
        source_mask = source_ids.ne(tokenizer.pad_token_id).type(torch.float)
        with torch.no_grad():
            task_id = model.get_task_id(eval_task)
            queries = model.get_inference_stats(input_ids=source_ids, labels=targets, task_id=task_id, phase='eval')
            queries = torch.nn.functional.normalize(queries, p=2.0, dim=1)
            task_sims.append(queries.detach().cpu())
    return torch.vstack(task_sims)

def get_reps_cosine_similarity(task_reps):
    task_avg_sim = torch.empty((len(task_reps), len(task_reps)))
    for i, treps_i in task_reps.items():
        for j, treps_j in task_reps.items():
            task_avg_sim[i,j] = treps_i.mm(treps_j.T).mean()
    return task_avg_sim

def main():
    t0 = time.time()
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    set_dist(args)
    seed_everything(args.seed)
    config, model, tokenizer = build_or_load_gen_model(args)
    datamodule = CodeXGlueDataModule(args, tokenizer)
    all_tasks = datamodule.all_tasks

    task_specific_params = ['num_train_epochs', 'learning_rate', 'patience', 'max_source_length',
                            'max_target_length', 'train_batch_size', 'eval_batch_size']
    args = get_task_arglist(args, task_specific_params, datamodule)

    if args.num_prompts_per_task > 0 or args.prompt_pool:
        print(f"***** Prompt Learning Rate: {args.prompt_lr} *****")
        model = PromptTuneT5(args, model, tokenizer, all_tasks)

    method, dims, name = args.projection_plot.split('-')
    if args.projection_plot == "":
        direc = f"/mnt/efs/people/ptky/project/incremental-learning/saved_runs/aws-debug_pool_replaypool_sep_proj~try=0"
        file = os.path.join(args.run_output_dir, f'checkpoints/best-bleu.bin')
        print(f"Reload best model from {file}")
        loaded_state_dict = torch.load(file)
        for buf in model._buffers.keys():
            model._buffers[buf] = torch.zeros(args.pool_size)
        model.load_state_dict(loaded_state_dict, strict=False)
    model.to(args.device)

    datamodule.setup(stage='fit')
    train_dataloaders = datamodule.train_dataloader()
    # val_dataloader = datamodule.val_dataloader()

    # Start training
    print(f"***** Get representations for {len(all_tasks)} tasks *****")
    print(f"Tasks: {all_tasks}\t")
    print(f"Eval Batch size: {args.eval_batch_size}\t")
    print(f"Max_src_lens: {args.max_source_length}\t")
    print(f"Max_tgt_lens: {args.max_target_length}\t")
    print("  " + "*" * 20)

    task_reps = ddict(list)
    for curr_idx, curr_task in enumerate(all_tasks):
        # Setup model, dataloader, and optimizer.
        train_loader = train_dataloaders[curr_task]
        task_base_train_bs = args.train_batch_size[curr_idx]
        task_max_source_length = args.max_source_length[curr_idx]
        task_max_target_length = args.max_target_length[curr_idx]
        print(f"Train Batch Size: {task_base_train_bs}\t BLEU eval Batch Size: {args.eval_batch_size[curr_idx]}")
        print(f"Max Src Len: {task_max_source_length}\t Max Tgt Len: {task_max_target_length}")
        print(f"Train: {len(train_loader) * train_loader.batch_size}")
        # print(f"Val: {len(val_dataloader[curr_task])*val_dataloader[curr_task].batch_size}")
        print(f"***** Eval results on Task {curr_task} *****".upper())
        task_reps[curr_idx] = get_task_queries(args, train_loader, model, tokenizer, eval_task=curr_task)
        print("\n\n")


    if args.projection_plot != '':
        reps = []
        targets = []
        for k,v in task_reps.items():
            reps.append(v)
            targets.append(torch.ones(v.shape[0]) * int(k))
        reps = torch.vstack(reps)
        targets = torch.cat(targets).numpy()

        keys = torch.FloatTensor(300, reps[0].size(-1)).uniform_() * args.uniform_scale

        if method =='tsne':
            print('Performing t-sne projection')
            meth = TSNE(n_components=int(dims), verbose=1, random_state=123)
        elif method == 'pca':
            print('Performing PCA projection')
            meth = PCA(n_components=int(dims))
        queries = meth.fit_transform(reps)
        keys_trans = meth.transform(keys)
        out = np.concatenate([queries, keys_trans])
        targets = np.concatenate([targets, np.ones(keys_trans.shape[0]) * (max(list(task_reps.keys()))+1)])

        if int(dims) == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(*zip(*out))
            plt.scatter(*zip(*out), c=targets, cmap='RdBu')
            # for ii in range(0,360,20):
            #     ax.view_init(elev=10., azim=ii)
            #     plt.savefig(f"./plots/figures/{args.projection_plot}_{ii}.png")
            # # plt.show()
            plt.savefig(f"./plots/figures/{args.projection_plot}.png")
        elif int(dims) == 2:
            df = pd.DataFrame()
            df["y"] = targets
            df["comp-1"] = out[:,0]
            df["comp-2"] = out[:,1]
            plot = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                            palette=sns.color_palette("hls", np.unique(targets).shape[0]),
                            data=df).set(title="CodeT5 queries for task")
            fig = plot[0].get_figure()
            fig.savefig(f"./plots/figures/{args.projection_plot}.png")
        else:
            raise ValueError(f"{dims }Dimension not supported.")

    task_rep_similarity = get_reps_cosine_similarity(task_reps)
    print(f"Task representation similarity: {task_rep_similarity.tolist()}\n\n")

    key_map = {
        "curr_count": "All",
        "concode_none_count": "CodeGen",
        "translate_java-cs_count": "CodeTrans",
        "summarize_ruby_count": "CodeSumm",
        "refine_small_count": "CodeRef",
    }
    print(f"'Prompt_id' : list(range(60)),")
    for k,v in model._buffers.items():
        if 'count' in k:
            print(f"'{key_map[k]}': {v.type(torch.int).tolist()},")

    print(f"\n\nFinished computing similarity on across all tasks: {get_elapse_time(t0)}".upper())


if __name__ == "__main__":
    main()

