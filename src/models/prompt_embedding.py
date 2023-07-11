# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

### This file is from ben and not used.

"""
Ben Athiwaratkun

This component is a prompt embedding used for prompt tuning
"""
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn import init
from typing import Optional
from torch.nn.parameter import Parameter

class EmbeddingWPrompts(nn.Module):
    # only support initializing with task names in the beginning
    # only supporting uniform prompt length
    __constants__ = ['num_embeddings', 'embedding_dim', 'padding_idx', 'max_norm',
                     'norm_type', 'scale_grad_by_freq', 'sparse']

    num_embeddings: int
    embedding_dim: int
    padding_idx: Optional[int]
    max_norm: Optional[float]
    norm_type: float
    scale_grad_by_freq: bool
    weight: Tensor
    sparse: bool

    def __init__(self, main_embeddings, tokenizer_token_len, task_names=[], n_prompt_tokens=0,
                 padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 sparse: bool = False, _weight: Optional[Tensor] = None
                 ):
        super(EmbeddingWPrompts, self).__init__()
        num_tasks = len(task_names)
        embedding_dim = main_embeddings.weight.size(1)
        self.num_embeddings = n_prompt_tokens*num_tasks + tokenizer_token_len
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.main_embeddings = main_embeddings

        ###################
        if n_prompt_tokens > 0:
            self.reset_prompt_embeddings(total_size=n_prompt_tokens*len(task_names))
        self.main_embeddings = Parameter(main_embeddings.weight[:tokenizer_token_len])
        self.tokenizer_token_len = tokenizer_token_len
        #print("tokenizer token len", self.tokenizer_token_len) #32100
        #print("embed weight size", main_embeddings.weight.size(0)) #32128
        assert self.tokenizer_token_len <= main_embeddings.weight.size(0), f"tokenizer token len {self.tokenizer_token_len} main embeddings size {main_embeddings.weight.size(0)}"
        self.n_prompt_tokens = n_prompt_tokens
        self.task_names = task_names
        assert len(task_names) > 0

        self.weight = torch.cat([self.main_embeddings, self.prompt_embeddings], dim=0)

    def reset_prompt_embeddings(self, total_size):
        self.prompt_embeddings = Parameter(Tensor(total_size, self.main_embeddings.weight.size(1)))
        main_std = torch.std(self.main_embeddings.weight)
        print(f"@ Prompt Embedding - initializing prompt embeddings. std = {main_std}")
        init.normal_(self.prompt_embeddings, std=float(main_std))

    def calculate_prompt_stats(self):
        prompt_mean = torch.mean(self.prompt_embeddings)
        prompt_std = torch.std(self.prompt_embeddings)
        print(f"Prompt embeddings mean = {prompt_mean} std = {prompt_std}")
        print(f"Main embeddings mean = {torch.mean(self.main_embeddings)} std = {torch.std(self.main_embeddings)}")

    def forward(self, input: Tensor) -> Tensor:
        #print(input)
        # and keep main and promp embs separately
        # need to look into how the output embeddings work
        self.weight = torch.cat([self.main_embeddings, self.prompt_embeddings], dim=0)
        return F.embedding(
            input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

    def add_tasks(self, task_names):
        assert False, 'Not yet supported'

    def reset_parameters(self) -> None:
        init.normal_(self.weight)
        self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def extra_repr(self) -> str:
        s = '{num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        return s.format(**self.__dict__)

    @classmethod
    def from_pretrained(cls, embeddings, freeze=True, padding_idx=None,
                        max_norm=None, norm_type=2., scale_grad_by_freq=False,
                        sparse=False):
        r"""Creates Embedding instance from given 2-dimensional FloatTensor.

        Args:
            embeddings (Tensor): FloatTensor containing weights for the Embedding.
                First dimension is being passed to Embedding as ``num_embeddings``, second as ``embedding_dim``.
            freeze (boolean, optional): If ``True``, the tensor does not get updated in the learning process.
                Equivalent to ``embedding.weight.requires_grad = False``. Default: ``True``
            padding_idx (int, optional): See module initialization documentation.
            max_norm (float, optional): See module initialization documentation.
            norm_type (float, optional): See module initialization documentation. Default ``2``.
            scale_grad_by_freq (boolean, optional): See module initialization documentation. Default ``False``.
            sparse (bool, optional): See module initialization documentation.

        Examples::

            >>> # FloatTensor containing pretrained weights
            >>> weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
            >>> embedding = nn.Embedding.from_pretrained(weight)
            >>> # Get embeddings for index 1
            >>> input = torch.LongTensor([1])
            >>> embedding(input)
            tensor([[ 4.0000,  5.1000,  6.3000]])
        """
        assert embeddings.dim() == 2, \
            'Embeddings parameter is expected to be 2-dimensional'
        rows, cols = embeddings.shape
        embedding = cls(
            num_embeddings=rows,
            embedding_dim=cols,
            _weight=embeddings,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse)
        embedding.weight.requires_grad = not freeze
        return embedding
