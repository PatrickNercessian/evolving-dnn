"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict

import torch
from torch.nn import functional as F

from torch.utils.data.dataloader import DataLoader
from mingpt.utils import CfgNode as CN

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
            elif torch.backends.mps.is_available():
                print("MPS available")
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

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
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
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
                break

if __name__ == "__main__":
    from mingpt.model import GPT
    from mingpt.trainer import Trainer
    from src.bpe import tokenize_string, VOCAB_SIZE

    import torch
    import os

    if os.path.exists('tokenized_data.pt'):
        data = torch.load('tokenized_data.pt')  # Save the tokenized data tensor to a file
    else:
        # Load, tokenize and save the input text
        with open('mingpt/input.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        data = tokenize_string(text)
        torch.save(data, 'tokenized_data.pt')

    # Create a simple dataset that generates sequences of block_size tokens
    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, data, block_size):
            self.data = data
            self.block_size = block_size

        def __len__(self):
            return len(self.data) - self.block_size

        def __getitem__(self, idx):
            # return a chunk of data and the next token as target
            x = self.data[idx:idx + self.block_size]
            y = self.data[idx + 1:idx + self.block_size + 1]
            return x, y

    # Initialize the model and training
    block_size = 128  # context length
    train_dataset = TextDataset(data, block_size)

    # Get default config and modify as needed
    config = GPT.get_default_config()
    config.model_type = None
    config.vocab_size = VOCAB_SIZE
    config.block_size = block_size
    config.n_layer = 2
    config.n_head = 2
    config.n_embd = 512

    # Create model
    model = GPT(config)

    # Training configuration
    train_config = Trainer.get_default_config()
    train_config.max_iters = 5000
    train_config.batch_size = 32
    train_config.learning_rate = 3e-4
    train_config.num_workers = 0
    
    trainer = Trainer(train_config, model, train_dataset)
    def batch_end_callback(trainer):
        if trainer.iter_num % 100 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()
