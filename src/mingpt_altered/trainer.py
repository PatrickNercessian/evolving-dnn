"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging
import time
from collections import defaultdict

import torch
from torch.nn import functional as F

from torch.utils.data.dataloader import DataLoader
from .utils import CfgNode as CN

class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        return C

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            # elif torch.backends.mps.is_available(): # TODO this doesn't work with adapative avg pooling having non-integer multiples
            #     print("MPS available")
            #     self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        logging.debug(f"running on device {self.device}")

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        # setup the dataloader
        # Check if we have an IterableDataset or a regular Dataset
        if hasattr(self.train_dataset, '__len__'):
            # Regular Dataset with RandomSampler
            train_loader = DataLoader(
                self.train_dataset,
                sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
                shuffle=False,
                pin_memory=True,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
            )
        else:
            # IterableDataset - no sampler needed
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=config.batch_size,
                pin_memory=True,
                num_workers=0,  # IterableDatasets work better with num_workers=0
            )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:

            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            logits = model(x)
            self.loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
            if math.isnan(self.loss):
                raise ValueError(f"Loss is nan at iter {self.iter_num}")

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                self.trigger_callbacks('on_train_end')
                break
