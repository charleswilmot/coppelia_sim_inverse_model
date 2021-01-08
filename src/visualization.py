import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import time
import os


plt.ion()


visualization_data_type = np.dtype([
    ("rewards", np.float32),
    ("target_return_estimates", np.float32),
    ("return_estimates", np.float32),
    ("critic_targets", np.float32),
    ("max_step_returns", np.float32),
])



class Visualization(object):
    def __init__(self, path, nx, ny, mini, maxi):
        self._fig = plt.figure()
        figManager = plt.get_current_fig_manager()
        figManager.full_screen_toggle()
        self._nx = nx
        self._ny = ny
        self._axs = [self._fig.add_subplot(nx, ny, i + 1) for i in range(nx * ny)]
        [ax.set_ylim([mini, maxi]) for ax in self._axs]
        self._lines = [defaultdict(lambda: None) for ax in self._axs]
        self._shown = False
        episode_length = int(os.path.basename(path).split('_')[0])
        with open(path, "rb") as f:
            self._data = np.frombuffer(f.read(), dtype=visualization_data_type)
        self._data = self._data.reshape((-1, episode_length))

    def update(self, index):
        for i, (lines, ax) in enumerate(zip(self._lines, self._axs)):
            data = self._data[index + i]
            ###
            if lines["rewards"]:
                x, = (data["rewards"] != 0).nonzero()
                y = data["rewards"][x]
                lines["rewards"].set_data(x, y)
            else:
                label = None if i else 'rewards'
                x, = (data["rewards"] != 0).nonzero()
                y = data["rewards"][x]
                lines["rewards"], = ax.plot(x, y, 'bo', label=label)
            ###
            if lines["target_return_estimates"]:
                lines["target_return_estimates"].set_ydata(data["target_return_estimates"])
            else:
                label = None if i else 'target_return_estimates'
                lines["target_return_estimates"], = ax.plot(
                    data["target_return_estimates"],
                    color='#ff0000',
                    label=label,
                )
            ###
            if lines["return_estimates"]:
                lines["return_estimates"].set_ydata(data["return_estimates"])
            else:
                label = None if i else 'return_estimates'
                lines["return_estimates"], = ax.plot(
                    data["return_estimates"],
                    color='#bb0000',
                    linestyle='--',
                    label=label,
                )
            ###
            if lines["critic_targets"]:
                lines["critic_targets"].set_ydata(data["critic_targets"])
            else:
                label = None if i else 'critic_targets'
                lines["critic_targets"], = ax.plot(
                    data["critic_targets"],
                    color='#990000',
                    label=label,
                )
            ###
            if lines["max_step_returns"]:
                lines["max_step_returns"].set_ydata(data["max_step_returns"])
            else:
                label = None if i else 'max_step_returns'
                lines["max_step_returns"], = ax.plot(
                    data["max_step_returns"],
                    color='#000000',
                    label=label,
                )
                if i == 0:
                    ax.legend()
            if not self._shown:
                self._fig.show()
                self._shown = True
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()

    def update_from_prompt(self):
        res = input("Jump to episode nb ? [0 - {}]   ".format(len(self._data)))
        if res == "":
            nb = len(self._data) - (self._nx * self._ny)
        else:
            nb = int(res)
        self.update(nb)

    def slideshow(self):
        for iteration in np.arange(len(self._data) - self._nx * self._ny, -1, -self._nx * self._ny):
            print(iteration)
            self.update(iteration)
            time.sleep(5)

    def __del__(self):
        plt.close(self._fig)


if __name__ == '__main__':
    import sys
    path = sys.argv[1]
    v = Visualization(path, 3, 1, -1.2, 1.2)
    v.slideshow()
    # while True:
    #     try:
    #         v.update_from_prompt()
    #     except KeyboardInterrupt:
    #         print("")
    #         break
