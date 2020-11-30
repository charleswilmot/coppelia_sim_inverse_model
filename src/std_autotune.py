import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax


class RingBuffer(object):
    def __init__(self, shape, dtype=np.float32):
        self._shape = shape
        self._dtype = dtype
        buffer_shape = np.copy(shape)
        self._midpoint = np.copy(buffer_shape[0])
        buffer_shape[0] *= 2
        self._buffer = np.zeros(shape=buffer_shape, dtype=dtype)
        self._current_reading_index = np.copy(self._midpoint)

    def _get_shape(self):
        return self._shape

    def _get_dtype(self):
        return self._dtype

    shape = property(_get_shape)
    dtype = property(_get_dtype)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._buffer[self._current_reading_index + key]
        reading_slice = slice(
            self._current_reading_index if key.start is None else self._current_reading_index + key.start,
            self._current_reading_index + self._midpoint if key.stop is None else self._current_reading_index + key.stop,
            key.step,
        )
        return self._buffer[reading_slice]

    def insert(self, data):
        self._current_reading_index -= 1
        if self._current_reading_index < 0:
            self._current_reading_index += self._midpoint
        self._buffer[self._current_reading_index] = data
        self._buffer[self._current_reading_index + self._midpoint] = data


def hann(x, center, size):
    return (0.5 * np.cos(2 * np.pi * (x - center) / size) + 0.5) * (np.abs(x - center) < size / 2).astype(np.float32)


class STDAutoTuner(object):
    def __init__(self, length, n_simulations, min_stddev, max_stddev, importance_ratio=1000):
        self._length = length
        self._n_simulations = n_simulations
        self._min_log_stddev = np.log(min_stddev)
        self._max_log_stddev = np.log(max_stddev)
        self._pair_dtype = np.dtype([('log_stddev', np.float32), ('reward', np.float32)])
        self._buffer = RingBuffer(shape=(length, n_simulations), dtype=self._pair_dtype)
        self._element = np.zeros(shape=(n_simulations,), dtype=self._pair_dtype)
        self._time_weights = np.zeros(shape=(length, n_simulations), dtype=np.float32)
        self._time_weights[:] = np.exp(np.linspace(
            np.log(1),
            np.log(1 / importance_ratio),
            length,
        ))[:, np.newaxis]
        self._convolved_since_last_insert = False
        self._convolved_since_last_insert_filter_size = None
        self._convolved_since_last_insert_n_evals = None

    def register_rewards(self, log_stddevs, rewards):
        self._element['log_stddev'] = log_stddevs
        self._element['reward'] = rewards
        self._buffer.insert(self._element)
        self._convolved_since_last_insert = False
        self._convolved_since_last_insert_filter_size = None
        self._convolved_since_last_insert_n_evals = None

    def init(self, log_stddev, reward):
        log_stddevs = np.full(shape=self._n_simulations, fill_value=log_stddev)
        rewards = np.full(shape=self._n_simulations, fill_value=reward)
        for i in range(self._length):
            self.register_rewards(log_stddevs, rewards)

    def time_weighted_convolve(self, filter_size, n_evals=500):
        if self._convolved_since_last_insert and \
                filter_size == self._convolved_since_last_insert_filter_size and \
                n_evals == self._convolved_since_last_insert_n_evals:
            return self._x, self._y, self._y_std
        data = self._buffer[:].flatten()
        args = np.argsort(data['log_stddev'])
        sorted_data = data[args]
        sorted_time_weights = self._time_weights.flatten()[args]

        def weighted_mean(log_stddev):
            args = np.where(np.logical_and(
                log_stddev - filter_size / 2 < sorted_data['log_stddev'],
                sorted_data['log_stddev'] < log_stddev + filter_size / 2,
            ))
            relevant_data = sorted_data[args]
            relevant_time_weights = sorted_time_weights[args]
            filter_weights = hann(relevant_data['log_stddev'], log_stddev, filter_size)
            weights = filter_weights * relevant_time_weights
            if np.sum(weights) == 0:
                return -10.0, 0.0
            else:
                y = np.average(relevant_data['reward'], weights=weights)
                y_std = np.sqrt(np.average((relevant_data['reward'] - y) ** 2, weights=weights))
                return y, y_std

        x = np.linspace(sorted_data['log_stddev'][0], sorted_data['log_stddev'][-1], n_evals)
        y = np.zeros(x.shape, dtype=np.float32)
        y_std = np.zeros(x.shape, dtype=np.float32)
        for i, log_stddev in enumerate(x):
            y[i], y_std[i] = weighted_mean(log_stddev)
        self._convolved_since_last_insert = True
        self._convolved_since_last_insert_filter_size = filter_size
        self._convolved_since_last_insert_n_evals = n_evals
        self._x = x
        self._y = y
        self._y_std = y_std
        return x, y, y_std

    def get_best_log_stddev(self, filter_size, std_importance, n_evals=500):
        x, y, y_std = self.time_weighted_convolve(filter_size, n_evals)
        where = np.logical_and(x <= self._max_log_stddev, x >= self._min_log_stddev)
        x = x[where]
        y = y[where]
        y_std = y_std[where]
        z = y + std_importance * y_std
        return x[np.argmax(z)]

    def get_log_stddevs(self, filter_size, std_importance, temperature=1.0, n_evals=500):
        x, y, y_std = self.time_weighted_convolve(filter_size, n_evals)
        where = np.logical_and(x < self._max_log_stddev, x > self._min_log_stddev)
        x = x[where]
        y = y[where]
        y_std = y_std[where]
        z = y + std_importance * y_std
        probs = softmax(z / temperature)
        return np.sort(np.random.choice(x, size=self._n_simulations, p=probs))

    def plot_rewards(self, ax, filter_size, std_importance, log_stddevs, n_evals=500):
        data = self._buffer[:].flatten()
        ax.plot(data["log_stddev"], data["reward"], 'bo', alpha=0.2)
        x, y, y_std = self.time_weighted_convolve(filter_size, n_evals)
        ax.plot(x, y, 'r-')
        ax.fill_between(x, y - y_std, y + y_std, color='g', alpha=0.2)
        ax.fill_between(x, y - std_importance * y_std, y + std_importance * y_std, color='g', alpha=0.3)
        for log_stddev in log_stddevs:
            ax.axvline(log_stddev, color='k', alpha=0.1)
        ax.set_xlabel("log stddev")
        ax.set_ylabel("reward")

    def save_plot(self, path, filter_size, std_importance, log_stddevs, n_evals=500):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.plot_rewards(ax, filter_size, std_importance, log_stddevs, n_evals)
        fig.savefig(path)
        plt.close(fig)


if __name__ == '__main__':
    rb = RingBuffer(shape=(5,))
    rb.insert(1)
    print(rb._buffer)
    print(rb[:])
    rb.insert(2)
    print(rb._buffer)
    print(rb[:])
    rb.insert(3)
    print(rb._buffer)
    print(rb[:])
    rb.insert(4)
    print(rb._buffer)
    print(rb[:])
    rb.insert(5)
    print(rb._buffer)
    print(rb[:])
    rb.insert(6)
    print(rb._buffer)
    print(rb[:])
    print("###")
    print(rb[:3])
    print(rb[3:])
    print(rb[2:4])

    import matplotlib.pyplot as plt

    at = STDAutoTuner(100, 10, importance_ratio=1000000)
    at.init(0.7, 3.0)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    at.plot_rewards(ax, filter_size=0.2, n_evals=500)
    plt.show()

    for i in range(123):
        log_stddevs = np.random.uniform(size=10)
        at.register_rewards(log_stddevs, log_stddevs ** 2 + np.random.uniform(size=10, low=-0.1, high=0.1))


    for i in range(123):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        at.plot_rewards(ax, filter_size=0.2, n_evals=500, current_low_log_stddev=0.2, current_high_log_stddev=0.5)
        plt.show()
        log_stddevs = np.random.uniform(size=10)
        at.register_rewards(log_stddevs, -log_stddevs ** 2 + np.random.uniform(size=10, low=-0.1, high=0.1))

    print(at._time_weights[:, 0])
