import pickle
import numpy as np


class Buffer(object):
    def __init__(self, size):
        self.size = size
        self.current_last = 0
        self._hparams = {
            "buffer_size": self.size,
        }
        self.dtype = None
        self.sample_index = 0

    def dump(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.buffer, f)

    def integrate(self, data):
        if self.dtype is None: # must create buffer and dtype
            self.dtype = data.dtype
            shape = np.copy(data.shape)
            shape[0] = self.size
            self.buffer = np.zeros(shape=shape, dtype=self.dtype)
        n = data.shape[0]
        indices = self.get_insertion_indices(n)
        if self.current_last < self.size:
            self.current_last += n
        if self.current_last > self.size:
            self.current_last = self.size
        self.buffer[indices] = data

    def get_insertion_indices(self, n):
        if self.current_last < self.size:
            space_remaining = self.size - self.current_last
            if space_remaining < n:
                # not enough room to insert the full episode
                part1 = np.random.choice(
                    np.arange(self.current_last),
                    n - space_remaining,
                    replace=False
                )
                part2 = np.arange(self.current_last, self.size)
                return np.concatenate((part1, part2))
            else: # enough empty space
                return slice(self.current_last, self.current_last + n)
        else: # buffer already full
            return np.random.choice(self.size, n, replace=False)

    def sample(self, batch_size):
        if self.current_last < batch_size or batch_size > self.size:
            return self.buffer[:self.current_last]
        batch_last = self.sample_index + batch_size
        if batch_last < self.current_last:
            ret = self.buffer[self.sample_index:batch_last]
            self.sample_index = batch_last
            return ret
        else: # enough data in buffer but exceed its size
            part1 = self.buffer[self.sample_index:self.current_last]
            part2 = self.buffer[:batch_last - self.current_last]
            self.sample_index = batch_last - self.current_last
            return np.concatenate((part1, part2))
