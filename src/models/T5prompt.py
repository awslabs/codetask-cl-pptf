# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0



### This file contains all the prompting related code for prompt pooling, TSPT, ShPT, PP + TF, etc

import copy
from multiprocessing.sharedctypes import Value
import torch
import torch.nn as nn
from collections import defaultdict as ddict
from tqdm import tqdm


class ProjectionLayer(nn.Module):
    def __init__(self, task_name, d_in, d_hid, d_out, dropout, affine=False):
        super(ProjectionLayer, self).__init__()
        self.task = task_name
        self.single_projection = nn.Sequential(
                                    nn.Linear(d_in, d_hid),
                                    nn.BatchNorm1d(d_hid, affine=affine),# track_running_stats=affine),
                                    nn.Tanh(),
                                    nn.Dropout(dropout),
                                    nn.Linear(d_hid, d_out),
                                    # nn.BatchNorm1d(d_out, affine=affine),# track_running_stats=affine),
                                )
        self.apply(self.init_weights)

    def forward(self, inputs, task_id=None):
        return self.single_projection(inputs)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

class TaskProjectionLayer(nn.Module):
    def __init__(self, task_names, d_in, d_hid, d_out, dropout, affine=True):
        super(TaskProjectionLayer, self).__init__()
        self.task_names = task_names
        proj_layer = [ProjectionLayer(task_name, d_in, d_hid, d_out, dropout, affine) for i, task_name in enumerate(task_names)]
        self.task_projection_list = nn.ModuleList(proj_layer)

    def forward(self, inputs, task_id):
        return self.task_projection_list[task_id](inputs)

class QueryFunction(nn.Module):
    def __init__(self, args, base_encoder, task_names, d_in, d_hid, d_out, dropout):
        super(QueryFunction, self).__init__()
        self.args = args
        self.task_names = task_names
        self.query_base_encoder = base_encoder
        if self.args.prompt_projection:
            if self.args.separate_projection:
                self.query_projection_layers = TaskProjectionLayer(self.task_names, d_in, d_hid, d_out, dropout, affine=True)
            else:
                self.query_projection_layers = ProjectionLayer('shared_projection', d_in, d_hid, d_out, dropout, affine=False)


    def forward(self, inputs, task_id):
        '''Unfinished as lower flexibility compared to the current implementation'''
        # self.query_base_encoder()
        if self.args.prompt_projection:
            if self.args.separate_projection:
                self.query_projection_layers
            else:
                pass
        return self.task_projection_list[task_id](inputs)


class PromptTuneT5(nn.Module):
    def __init__(self, args, model, tokenizer, datamodule):
        super(PromptTuneT5, self).__init__()
        self.args = args
        self.code_model = model
        self.tokenizer = tokenizer
        self.datamodule = datamodule
        self.tasks = datamodule.all_tasks
        self.num_tasks = len(self.tasks)
        self.prompt_method = self.args.prompt_method
        self.prompt_loss_type = self.args.prompt_loss_type
        self.decoder_start_token_id_use = self.code_model.config.decoder_start_token_id

        if 'pool' in self.prompt_method:
            if self.args.task_specific_ratio != 0:
                self.num_task_specific_prompts = int(self.args.num_pool_prompt_tokens * self.args.num_prompts_per_task * self.args.task_specific_ratio)
                self.num_task_specific_keys = self.num_task_specific_prompts// self.args.num_pool_prompt_tokens
                self.total_task_specific_prompts = self.num_task_specific_prompts * len(self.tasks)
                self.total_task_specific_keys = self.num_task_specific_keys * len(self.tasks)
                self.top_k = self.args.num_prompts_per_task - self.num_task_specific_keys
                self.args.pool_size = self.args.org_pool_size - self.total_task_specific_keys
            else:
                self.num_task_specific_prompts = 0
                self.num_task_specific_keys = 0
                self.total_task_specific_prompts = 0
                self.total_task_specific_keys = 0
                self.top_k = self.args.num_prompts_per_task - self.num_task_specific_keys

            if self.args.pool_teacher != '':
                self.task_pairs = (self.num_tasks * ((self.num_tasks-1))) // 2
                self.shared_keys_per_task = (self.num_tasks - 1) * self.args.num_shared_keys_per_pair
                self.total_shared_keys = self.task_pairs * self.args.num_shared_keys_per_pair
                self.num_task_keys = self.args.num_prompts_per_task - self.shared_keys_per_task
                self.total_task_keys = self.num_task_keys * self.num_tasks
                print(f"Shared keys per task: {self.shared_keys_per_task}\tTask keys per task: {self.num_task_keys}")
                self.args.pool_size = self.total_shared_keys + self.total_task_keys

            self.num_prompts = self.args.pool_size * self.args.num_pool_prompt_tokens
            assert self.args.pool_size > self.top_k, f"pool size smaller than per task prompt"
            if self.args.pool_freq_norm or self.args.pool_freq:
                self.register_buffer(f'curr_count', torch.ones(self.args.pool_size) * 1e-8)
                for i, t in enumerate(self.tasks):
                    self.register_buffer(f'{t}_count', torch.ones(self.args.pool_size) * 1e-8)


        elif self.prompt_method == 'tspt':
            self.num_prompts = self.args.num_prompts_per_task * self.num_tasks
        elif self.prompt_method == 'shpt':
            self.num_prompts = self.args.num_prompts_per_task
        else:
            raise ValueError(f"Prompt method {self.prompt_method} not implemented!")

        if self.args.num_prompts_per_task > 0 and 'pool' not in self.prompt_method:
            self.prompt_embeddings = self.initialize_embedding(n_prompts=self.num_prompts, out_dim=self.code_model.config.d_model, prompt_init='vocab')

        if 'pool' in self.prompt_method:
            self.query_base_model = copy.deepcopy(self.code_model.encoder)
            if self.args.prompt_projection:
                if self.args.separate_projection:
                    self.query_projection_layers = TaskProjectionLayer(self.tasks, self.code_model.config.d_model, self.args.projection_hid_dim, self.args.projection_out_dim, self.args.dropout)
                else:
                    self.query_projection_layers = ProjectionLayer('shared_projection', self.code_model.config.d_model, self.args.projection_hid_dim, self.args.projection_out_dim, self.args.dropout)

    def initialize_prompt_pool(self, load=False):
        out_dim = self.args.projection_out_dim if self.args.prompt_projection else self.code_model.config.d_model
        if load:
            self.pool_keys = nn.parameter.Parameter(torch.zeros(self.args.pool_size, out_dim))
            self.prompt_embeddings = nn.parameter.Parameter(torch.zeros(self.num_prompts, out_dim))

            if self.args.task_specific_ratio != 0:
                self.prompt_tasks = nn.parameter.Parameter(torch.zeros((self.total_task_specific_prompts, out_dim)))
            for ti in range(len(self.tasks)):
                self.register_buffer(f'task2key_{ti}', torch.tensor([]))

        else:
            if self.args.prompt_init == 'data' and self.args.prompt_key_init == 'data':
                if self.args.task_specific_ratio != 0:
                    print('Inititalizing task specific prompts!')
                    _, prompts, _, task2prompt, _, _ = self.data_init_key_prompts(self.args, self.tokenizer, self.datamodule, self.total_task_specific_prompts)
                    # self.task_prompts = nn.ParameterList([nn.parameter.Parameter(prompts[task2prompt[ti]]) for ti, task in enumerate(self.tasks)])
                    # self.register_parameter(name=f'prompt_tasks', param=nn.parameter.Parameter(prompts))
                    self.prompt_tasks = nn.parameter.Parameter(prompts)
                    # for ti, task in enumerate(self.tasks):
                    #     self.register_parameter(name=f'prompt_task_{ti}', param=nn.parameter.Parameter(prompts[task2prompt[ti]]))

                keys, prompts, _, _, _, task2key = self.data_init_key_prompts(self.args, self.tokenizer, self.datamodule, self.num_prompts)
                if self.args.pool_teacher != '':
                    # self.task_pairs = (self.num_tasks * ((self.num_tasks-1))) // 2
                    # self.shared_keys_per_task = (self.num_tasks - 1) * self.args.num_shared_keys_per_pair
                    # self.total_shared_keys = self.task_pairs * self.args.num_shared_keys_per_pair
                    # self.num_task_keys = self.args.num_prompts_per_task - self.shared_keys_per_task
                    # self.total_task_keys = self.num_task_keys * self.num_tasks
                    # print(f"Shared keys per task: {self.shared_keys_per_task}\tTask keys per task: {self.num_task_keys}")
                    assignments = ddict(list)
                    leftover = []
                    for ti in range(self.num_tasks):
                        assignments[ti] += task2key[ti][:self.num_task_keys]
                        leftover += task2key[ti][self.num_task_keys:]
                        del task2key[ti][:self.num_task_keys]
                    for ii in range(self.num_tasks):
                        for jj in range(self.num_tasks):
                            if jj <= ii:
                                continue
                            assignments[ii] += leftover[:self.args.num_shared_keys_per_pair]
                            assignments[jj] += leftover[:self.args.num_shared_keys_per_pair]
                            del leftover[:self.args.num_shared_keys_per_pair]
                    # used_key_indices = []
                    # used_prompt_indices = []
                    # for v in assignments.values():
                    #     used_key_indices += v
                    # keys = keys[used_key_indices]
                    # for k in used_key_indices:
                    #     used_prompt_indices += range()
                    # keys = keys[used_key_indices]
                    pass

                for k,v in task2key.items():
                    self.register_buffer(f'task2key_{k}', torch.tensor(assignments[k]))
                self.pool_keys = nn.parameter.Parameter(keys)
                self.prompt_embeddings = nn.parameter.Parameter(prompts)
            else:
                self.prompt_embeddings = self.initialize_embedding(n_prompts=self.num_prompts, out_dim=self.code_model.config.d_model, prompt_init=self.args.prompt_init)
                self.pool_keys = self.initialize_embedding(n_prompts=self.args.pool_size, out_dim=out_dim, prompt_init=self.args.prompt_key_init)

            if self.args.prompt_loss_type == "clus_center":
                self.register_buffer('prompt_init_keys', self.pool_keys.clone().detach())
                for task_id, task_name in enumerate(self.tasks):
                    self.register_buffer(f'{task_id}_key_center', self.prompt_init_keys[self._buffers[f'task2key_{task_id}']].mean(dim=0).detach())

    def get_parameter_by_name(self, name):
        for n, p in self.named_parameters():
            if n == name:
                return p
        raise ValueError(f"parameter named {name} not found!!")

    def initialize_embedding(self, n_prompts=None, out_dim=None, prompt_init=None):
        if prompt_init == 'vocab':
            embeddings = self.code_model.get_input_embeddings()
            indices = torch.randperm(embeddings.weight.size(0))[:n_prompts]
            return nn.parameter.Parameter(embeddings.weight[indices].clone().detach())
        elif prompt_init == 'uniform':
            return nn.parameter.Parameter(torch.FloatTensor(n_prompts, out_dim).uniform_() * self.args.uniform_scale)
        elif prompt_init == "xavier":
            keys = torch.FloatTensor(n_prompts, out_dim)
            torch.nn.init.xavier_normal_(keys)
            keys = keys * self.args.uniform_scale * 10
            return nn.parameter.Parameter(keys)
        elif prompt_init == "zeros":
            keys = torch.zeros(n_prompts, out_dim)
            return nn.parameter.Parameter(keys)
        else:
            NotImplementedError()

    def get_pool_prompt_indices(self, top_pool_indices):
        ind_list = []
        for batch_ind in top_pool_indices:
            start_ind = batch_ind * self.args.num_pool_prompt_tokens
            end_ind = (batch_ind+1) * self.args.num_pool_prompt_tokens
            ind_list.append(torch.cat([torch.arange(si, ei) for si, ei in zip(start_ind, end_ind)]))
        return torch.stack(ind_list)

    def add_task_tags(self, input_ids, task_name):
        task_tokens = self.tokenizer.encode([f"<{task_name.split('_')[0]}>", f"<{task_name.split('_')[1]}>"], return_tensors='pt')
        task_tokens = task_tokens[:, :-1].repeat(input_ids.size(0), 1).to(self.args.device)
        encoder_input = torch.cat([task_tokens, input_ids[:, 1:]], dim=1)
        return encoder_input

    def get_scores(self, queries, keys, phase, type='cosine'):
        if type == 'cosine':
            queries = torch.nn.functional.normalize(queries, p=2.0, dim=1)
            keys = torch.nn.functional.normalize(keys, p=2.0, dim=1)
            cos_sim = queries.mm(keys.T)
        else:
            raise NotImplementedError(f"Score type: {type} not implemented!!")
        if self.args.pool_freq_norm and phase == 'train':
            curr_freq = torch.nn.functional.normalize(self._buffers['curr_count'], p=1, dim=0)
            scores = cos_sim * (1 / curr_freq.unsqueeze(0))
            return cos_sim, scores

        assert (cos_sim>=-1).sum() == cos_sim.numel(), f'cosine similarity below -1'
        assert (cos_sim<=1).sum() == cos_sim.numel(), f'cosine similarity over 1'

        return cos_sim, cos_sim

    def pool_embeddings(self, embeds, attention, pool_mode='mean'):
        if pool_mode == 'sos':
            pooled_output = embeds.detach()[:,0,:]
        elif pool_mode == 'mean':
            pooled_output = (torch.sum(embeds * attention.unsqueeze(-1), dim=1) / attention.sum(dim=1, keepdims=True)).detach()
        else:
            raise NotImplementedError()
        return pooled_output

    def get_queries(self, input_ids, targets, task_id):
        encoder_input = input_ids
        if self.args.io_queries:
            encoder_input = torch.cat([encoder_input, targets[:,1:]], dim=1)
        encoder_attention = encoder_input.ne(self.tokenizer.pad_token_id).type(torch.float).to(self.args.device)
        last_hidden_state = self.query_base_model(encoder_input, encoder_attention)['last_hidden_state']
        queries = self.pool_embeddings(last_hidden_state, encoder_attention, self.args.query_pooling_mode)
        if self.args.prompt_projection:
            queries = self.query_projection_layers(queries, task_id)
        return queries

    def get_topk(self, similarity, scores, k, task_id, phase, batched=False):
        if batched:
            top_scores, top_indices = torch.topk(scores, k=self.top_k, dim=1, sorted=True)
            batch_top_indices, batch_freq = torch.unique(top_indices.flatten(), return_counts=True)
            _, ind = torch.topk(batch_freq, k=self.top_k, sorted=True)
            top_indices = batch_top_indices[ind]
            top_indices = top_indices.repeat(scores.size(0),1)
            top_vals = torch.cat([a[i].unsqueeze(0) for a, i in zip(similarity, top_indices)])
        else:
            if (self.args.pool_teacher == 'train' and phase == 'train') or self.args.pool_teacher == 'both':
                top_indices = self._buffers[f"task2key_{task_id}"].repeat(scores.size(0),1)
            else:
                top_scores, top_indices = torch.topk(scores, k=k, dim=1, sorted=True)
            top_vals = torch.cat([a[i].unsqueeze(0) for a, i in zip(similarity, top_indices)])
        return top_vals, top_indices

    def get_prompts_indices(self, input_ids, labels, task_id, phase):
        if self.prompt_method == 'pool':
            queries = self.get_queries(input_ids, labels, task_id)
            similarity, scores = self.get_scores(queries, self.pool_keys, phase, type='cosine')
            top_vals, top_pool_indices = self.get_topk(similarity, scores, k=self.top_k, task_id=task_id, phase=phase, batched=self.args.batched_prompts)
            prompt_indices = self.get_pool_prompt_indices(top_pool_indices)
            prompt_vectors = self.prompt_embeddings[prompt_indices]

            if self.args.task_specific_ratio != 0:
                # task_prompts = self.get_parameter_by_name(name=f'prompt_task_{task_id}').repeat(input_ids.size(0), 1, 1)
                # task_prompts = self.get_parameter(f'prompt_task_{task_id}').repeat(input_ids.size(0), 1, 1)
                task_prompts = self.prompt_tasks[task_id : task_id + self.num_task_specific_prompts, :].repeat(input_ids.size(0), 1, 1)
                # task_prompts = self.task_prompts[task_id].repeat(input_ids.size(0), 1, 1)
                prompt_vectors = torch.cat([task_prompts, prompt_vectors], dim=1)

            if (self.args.pool_freq_norm or self.args.pool_freq) and phase == 'train':
                pid, pfreq = torch.unique(top_pool_indices.reshape(-1).clone().detach(), return_counts=True)
                self._buffers[f'{self.tasks[task_id]}_count'][pid] += pfreq
            return prompt_vectors, top_pool_indices, top_vals

        elif self.prompt_method == 'tspt':
            indices = torch.arange(task_id * self.args.num_prompts_per_task, (task_id + 1) * self.args.num_prompts_per_task, dtype=int)
            indices = indices.repeat(input_ids.size(0),1)
            prompt_vectors = self.prompt_embeddings[indices]
            return prompt_vectors, indices, None

        elif self.prompt_method == 'shpt':
            prompt_vectors = self.prompt_embeddings.repeat(input_ids.size(0), 1, 1)
            return prompt_vectors, None, None

        elif self.prompt_method == 'pool_fixed':
            queries = self.get_queries(input_ids, labels, task_id)
            similarity, scores = self.get_scores(queries, self.pool_keys, phase, type='cosine')
            similarity = similarity[:, self._buffers[f'task2key_{task_id}']]
            scores = scores[:, self._buffers[f'task2key_{task_id}']]
            top_scores, top_pool_indices = torch.topk(scores, k=self.top_k, dim=1, sorted=True)
            top_vals = torch.cat([a[i].unsqueeze(0) for a, i in zip(similarity, top_pool_indices)])
            top_pool_indices = top_pool_indices + self._buffers[f'task2key_{task_id}'][0]
            prompt_indices = self.get_pool_prompt_indices(top_pool_indices)
            prompt_vectors = self.prompt_embeddings[prompt_indices]
            if (self.args.pool_freq_norm or self.args.pool_freq) and phase == 'train':
                pid, pfreq = torch.unique(top_pool_indices.reshape(-1).clone().detach(), return_counts=True)
                self._buffers[f'{self.tasks[task_id]}_count'][pid] += pfreq
            return prompt_vectors, top_pool_indices, top_vals
        else:
            pass

    def add_prompt_to_input(self, input_ids, attention_mask, labels, task_id, phase):
        if task_id == -1:
            input_ids = self.add_task_tags(input_ids, 'refine_small')
        else:
            input_ids = self.add_task_tags(input_ids, self.tasks[task_id])
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).to(self.args.device)
        input_embed_part = self.code_model.encoder.embed_tokens(input_ids)
        prompt_vectors, prompt_indices, prompt_scores = self.get_prompts_indices(input_ids, labels, task_id, phase)
        final_input_ids = torch.cat([prompt_vectors, input_embed_part], 1)
        mask_prompt = torch.ones_like(prompt_vectors[:,:,0]).to(self.args.device)
        final_attention_mask = torch.cat([mask_prompt, attention_mask], dim=1)
        prompt_loss = self.get_prompt_loss(prompt_vectors, prompt_indices, prompt_scores, task_id)
        return final_input_ids, final_attention_mask, prompt_loss

    def get_task_id(self, task):
        if isinstance(task, int):
            task_id = task
        elif isinstance(task, str):
            try:
                task_id = self.tasks.index(task)
            except:
                task_id = -1
        else:
            task_id = None
        return task_id

    def get_inference_stats(self, input_ids, labels, task_id, phase):
        input_ids = self.add_task_tags(input_ids, self.tasks[task_id])
        if 'pool' in self.prompt_method:
            queries = self.get_queries(input_ids, labels, task_id)
            if (self.args.pool_freq_norm or self.args.pool_freq) and phase != 'init':
                similarity, scores = self.get_scores(queries, self.pool_keys, phase, type='cosine')
                top_vals, top_pool_indices = self.get_topk(similarity, scores, batched=self.args.batched_prompts)
                pid, pfreq = torch.unique(top_pool_indices.reshape(-1).clone().detach(), return_counts=True)
                self._buffers[f'{self.tasks[task_id]}_count'][pid] += pfreq
            return queries

    def get_prompt_loss(self, prompt_vectors, prompt_indices, prompt_scores, task_id):
        if prompt_scores is None:
            return torch.tensor([0]).type(torch.float).to(self.args.device)

        prompt_loss = -1 * self.args.pool_lambda * prompt_scores.sum(dim=1)
        if self.prompt_loss_type == "clus_center":
            center_loss = (self.pool_keys[prompt_indices] - self._buffers[f'{task_id}_key_center'].repeat(1,1,1)).pow(2).sum(dim=2).sqrt().mean(dim=1)
            center_loss = self.args.center_lambda * center_loss
            prompt_loss = prompt_loss + center_loss

        return prompt_loss

    def forward(self, input_ids, attention_mask, labels, decoder_attention_mask, task_name=None, phase=None):
        task_id = self.get_task_id(task_name)
        if task_id is None:
            return self.code_model(input_ids, attention_mask, labels, decoder_attention_mask)
        else:
            final_input_ids, final_attention_mask, prompt_loss = self.add_prompt_to_input(input_ids, attention_mask, labels, task_id, phase)
            labels[labels.eq(self.tokenizer.pad_token_id)] = -100
            outputs  = self.code_model(inputs_embeds=final_input_ids, attention_mask=final_attention_mask,
                                labels=labels, decoder_attention_mask=decoder_attention_mask)
            return outputs, prompt_loss

    def generate(self, input_ids, attention_mask, labels, task_name, kwargs):
        task_id = self.get_task_id(task_name)
        phase = 'eval'
        if task_id is None:
            return self.code_model.generate(input_ids, attention_mask, **kwargs)
        else:
            final_input_ids, final_attention_mask, prompt_scores = self.add_prompt_to_input(input_ids, attention_mask, labels, task_id, phase)
            decoder_input_ids = ( torch.ones((input_ids.shape[0], 1), dtype=torch.long, device=self.args.device) * self.decoder_start_token_id_use )

            return self.code_model.generate(
                inputs_embeds=final_input_ids,
                decoder_input_ids=decoder_input_ids,
                attention_mask=final_attention_mask,
                **kwargs
            )

    def get_task_queries(self, args, dataloader, tokenizer, eval_task=None):
        self.code_model.eval()
        keys, prompts = [], []
        for batch in tqdm(dataloader, total=len(dataloader), desc="Prompt Init"):
            batch = tuple(t.to(self.args.device) for t in batch)
            source_ids, target_ids = batch
            target_ids = target_ids if args.io_queries else None
            source_mask = source_ids.ne(tokenizer.pad_token_id).type(torch.float)
            with torch.no_grad():
                task_id = self.get_task_id(eval_task)
                input_embeds = self.code_model.encoder.embed_tokens(source_ids)
                batch_prompts = self.pool_embeddings(input_embeds, source_mask, pool_mode='mean')
                batch_keys = self.get_inference_stats(source_ids, target_ids, task_id, phase='init')

                prompts.append(batch_prompts.detach().cpu())
                keys.append(batch_keys.detach().cpu())
        return torch.vstack(keys), torch.vstack(prompts)

    def data_init_key_prompts(self, args, tokenizer, datamodule, n_prompts):
        query_key_init_dataloaders = self.datamodule.get_key_prompt_init_dataloaders(n_prompts)
        all_keys, all_prompts, prompt_task_ids = [], [], []
        for ti, task, in enumerate(datamodule.all_tasks):
            keys, prompts = self.get_task_queries(args, query_key_init_dataloaders[task], tokenizer, eval_task=task)
            prompt_task_ids.append(torch.ones(keys.size(0)) * ti)
            # key_task_ids.append(torch.ones((1)) * ti)
            all_keys.append(torch.nn.functional.normalize(keys, p=2.0, dim=1))
            all_prompts.append(torch.nn.functional.normalize(prompts, p=2.0, dim=1))
        all_keys = torch.vstack(all_keys)
        prompt_task_ids = torch.cat(prompt_task_ids)
        keys = []
        key_task_ids = []
        for i, pi in enumerate(range(0, len(all_keys), self.args.num_pool_prompt_tokens)):
            key_task_ids.append((i, prompt_task_ids[pi].item()))
            if self.args.keys_agg == "mean":
                keys.append(all_keys[pi: pi + self.args.num_pool_prompt_tokens].mean(dim=0))
            elif self.args.keys_agg == "first":
                keys.append(all_keys[pi])
            elif self.args.keys_agg == "random":
                ind = torch.randint(self.args.num_pool_prompt_tokens,(1,))
                keys.append(all_keys[pi+ind])
            else:
                raise NotImplementedError(f"Key aggregation not Implemented!")
        keys = torch.vstack(keys)
        prompts = torch.vstack(all_prompts)
        prompt2task = {i:t.type(torch.int).item() for i, t in enumerate(prompt_task_ids)}
        key2task = {ki:int(t) for ki, t in key_task_ids}
        task2prompt = ddict(list)
        task2key = ddict(list)
        for i, t in prompt2task.items():
            task2prompt[t].append(i)
        for ki, t in key2task.items():
            task2key[t].append(ki)
        return keys, prompts, prompt2task, task2prompt, key2task, task2key



