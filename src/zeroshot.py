import os, sys
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
import matplotlib
import pdb
import random
import multiprocessing

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
# from utils.configs import *
from utils.replay import Buffer
from models.T5prompt import PromptTuneT5
from dataloaders.generation_loader import CodeXGlueDataModule
from cont_gen import *


def val_bleu_epoch(args, val_dataloader, val_examples, model, tokenizer, max_target_length, curr_task, eval_task, fname):
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
            print("Save the predictions into %s", output_fn)
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

    print(f"Train Task: {curr_task} Eval Task: {eval_task} \t " + "  ".join(f'{k}={v:.4f}' for k,v in sorted(results.items())))
    return results


def load_model(args, model, tokenizer, datamodule, run_path, criteria):
    if args.num_prompts_per_task > 0 or args.prompt_method == 'pool':
        model = PromptTuneT5(args, model, tokenizer, datamodule)
    if args.prompt_method == 'pool':
        model.initialize_prompt_pool(load=True)
    model.to(args.device)
    print('INITIALIZED DATALOADER AND MODEL')

    file = os.path.join(args.run_output_dir, f'{run_path}/checkpoints/{criteria}.bin')
    loaded_state_dict = torch.load(file)
    for buf in model._buffers.keys():
        model._buffers[buf] = torch.empty_like(loaded_state_dict[buf])
    missing_keys, unexpected_keys = model.load_state_dict(loaded_state_dict, strict=False)
    print(f"Missing keys: {missing_keys}\tUnexpected Keys: {unexpected_keys}")
    return model



def main():
    t0 = time.time()
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    set_dist(args)
    seed_everything(args.seed)


    ### INITIALIZE DATALOADER AND MODEL
    config, model, tokenizer = build_or_load_gen_model(args)
    args.stream = 'concode_none,translate_java-cs,summarize_ruby'
    datamodule = CodeXGlueDataModule(args, tokenizer)
    # datamodule.setup(stage='test')
    run_path = "/mnt/efs/people/ptky/project/incremental-learning/saved_runs/aws-teacher_tune1/pool_teacher_train_plr0.1_ER5k~try=0"
    criteria = 'summarize_ruby'
    model = load_model(args, model, tokenizer, datamodule, run_path, criteria)

    args.stream = 'concode_none,translate_java-cs,summarize_ruby,refine_small'
    task_specific_params = ['num_train_epochs', 'learning_rate', 'patience', 'max_source_length',
                            'max_target_length', 'train_batch_size', 'eval_batch_size']
    datamodule_all = CodeXGlueDataModule(args, tokenizer)
    args = get_task_arglist(args, task_specific_params, datamodule_all)
    datamodule_all.setup(stage='test')
    test_examples, test_data, test_dataloader = datamodule_all.test_dataloader()

    test_task = 'refine_small'
    test_idx = 3
    result = val_bleu_epoch(args, test_dataloader[test_task], test_examples[test_task], model, tokenizer, args.max_target_length[test_idx], 'none', test_task, criteria)



if __name__ == "__main__":
    main()