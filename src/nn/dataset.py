import torch
from torch.utils.data import IterableDataset

class HuggingFaceIterableDataset(IterableDataset):  # TODO we should store on disk when we pull, and load from disk if it's there
    def __init__(self, iterable_dataset, tokenizer, block_size, max_samples=None):
        """
        Wrapper for HuggingFace iterable datasets to work with PyTorch DataLoader
        
        Args:
            iterable_dataset: HuggingFace IterableDataset instance
            tokenizer: Tokenizer to use for encoding text
            block_size: Sequence length for the model
            max_samples: Maximum number of samples to process (None for unlimited)
        """
        self.iterable_dataset = iterable_dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.max_samples = max_samples
        
    def __iter__(self):
        samples_processed = 0
        buffer = []
        
        for example in self.iterable_dataset:
            if self.max_samples and samples_processed >= self.max_samples:
                break
                
            # Tokenize the text
            text = example.get("text", "")
            if not text:
                continue
                
            encoded = self.tokenizer.encode(text)
            tokens = encoded.ids
            
            # Add tokens to buffer
            buffer.extend(tokens)
            
            # Yield sequences from the buffer
            while len(buffer) >= self.block_size + 1:
                # Extract input and target sequences
                x = torch.tensor(buffer[:self.block_size], dtype=torch.long)
                y = torch.tensor(buffer[1:self.block_size + 1], dtype=torch.long)
                
                yield x, y
                
                # Remove processed tokens from buffer
                buffer = buffer[self.block_size:]
                samples_processed += 1
                
                if self.max_samples and samples_processed >= self.max_samples:
                    return

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