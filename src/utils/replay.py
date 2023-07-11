# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import torch
import numpy as np
from typing import Tuple
from collections import defaultdict as ddict
from copy import deepcopy


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size


class Buffer:
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, buffer_size, device, n_tasks=None, mode='reservoir'):
        assert mode in ['ring', 'reservoir', 'task-balanced']
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.functional_index = eval(mode)
        self.mode = mode
        if self.mode == 'ring':
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        if self.mode == 'task-balanced':
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        self.attributes = ['examples', 'labels', 'logits', 'task_labels']
        self.task2idx = ddict(list)


    def to(self, device):
        self.device = device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, getattr(self, attr_str).to(device))
        return self


    def __len__(self):
        return min(self.num_seen_examples, self.buffer_size)


    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     logits: torch.Tensor, task_labels: torch.Tensor) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.float32 if attr_str.endswith('its') else torch.int64
                setattr(self, attr_str, torch.zeros((self.buffer_size,*attr.shape[1:]), dtype=typ))


    def add_data(self, examples, labels=None, logits=None, task_labels=None):
        """
        Adds the data to the memory buffer according to the strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :return:
        """
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels)

        for i in range(examples.shape[0]):
            if self.mode == 'reservoir':
                index = reservoir(self.num_seen_examples, self.buffer_size)
            elif self.mode == 'ring' and task_labels is not None:
                index = ring(self.num_seen_examples, self.buffer_portion_size, task_labels[i])
            # elif self.mode == 'task-balanced':
            #     # index = ring(self.num_seen_examples, self.buffer_portion_size, self.task_number)
            #     pass
            else:
                raise NotImplementedError(f"Mode {self.mode} is not implemented.")

            self.num_seen_examples += 1
            if index >= 0:
                self.examples[index] = examples[i].detach().cpu()
                if labels is not None:
                    self.labels[index] = labels[i].detach().cpu()
                if logits is not None:
                    self.logits[index] = logits[i].detach().cpu()
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].detach().cpu()
                self.task2idx[task_labels[i].item()].append(index.item())


    def get_data(self, size: int, transform=None, return_index=False) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if self.mode == 'ring':
            populated_portion_length = (self.labels != -1).sum().item()
            if size > populated_portion_length:
                size = populated_portion_length
            choice = np.random.choice(populated_portion_length, size=size, replace=False)
        elif self.mode == 'reservoir':
            if size > min(self.num_seen_examples, self.examples.shape[0]):
                size = min(self.num_seen_examples, self.examples.shape[0])
            choice = np.random.choice(min(self.num_seen_examples, self.examples.shape[0]), size=size, replace=False)
        else:
            raise NotImplementedError(f"Mode {self.mode} is not implemented.")

        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee) for ee in self.examples[choice]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)

        if not return_index:
            return ret_tuple
        else:
            return (torch.tensor(choice).to(self.device), ) + ret_tuple


    def get_data_by_index(self, indexes, transform=None) -> Tuple:
        """
        Returns the data by the given index.
        :param index: the index of the item
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee) for ee in self.examples[indexes]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str).to(self.device)
                ret_tuple += (attr[indexes],)
        return ret_tuple


    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.mode == 'ring':
            if self.num_seen_examples == 0 and self.task_number == 0:
                return True
            else:
                return False
        elif self.mode == 'reservoir':
            if self.num_seen_examples == 0:
                return True
            else:
                return False
        else:
            raise NotImplementedError(f"Mode {self.mode} is not implemented.")


    def get_all_data(self, transform=None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee) for ee in self.examples]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple


    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0

