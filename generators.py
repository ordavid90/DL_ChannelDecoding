import numpy as np


class InfoGenerator:
    def __init__(self, info_len, batch_size=32, shuffle=True):
        self.info_len = info_len
        self.batch_size = batch_size
        self.indices = np.arange(2**info_len)
        self.shuffle = shuffle

    def on_epoch_end(self):
        self.indices = np.arange(2 ** self.info_len)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, batch_inds):
        X = np.empty((self.batch_size, self.info_len), dtype=float)
        for i, ind in enumerate(batch_inds):
            X[i,] = np.array([int(b) for b in ('{0:0' + str(self.info_len) + 'b}').format(ind)])
        Y = np.copy(X)
        return X, Y

    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, batch_num):
        batch_inds = self.indices[batch_num * self.batch_size: (batch_num + 1) * self.batch_size]
        X, Y = self.__data_generation(batch_inds)
        return X, Y










