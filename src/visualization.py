import matplotlib.pyplot as plt


plt.ion()

class Visualization(object):
    def __init__(self, mini, maxi):
        self._fig = plt.figure()
        self._critic_prediction_ax = self._fig.add_subplot(111)
        self._critic_prediction_ax.set_ylim([mini, maxi])
        self._target_line = None

    def __call__(self, target, prediction):
        if self._target_line is None:
            self._target_line, = self._critic_prediction_ax.plot(
                target, label='target'
            )
            self._prediction_line, = self._critic_prediction_ax.plot(
                prediction, label='prediction'
            )
            self._critic_prediction_ax.legend()
            self._fig.show()
            self._fig.canvas.flush_events()
        else:
            self._target_line.set_ydata(target)
            self._prediction_line.set_ydata(prediction)
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()

    def __del__(self):
        plt.close(self._fig)


if __name__ == '__main__':
    import time

    v = Visualization()
    v([1,2,3], [4, 5, 6])
    time.sleep(1)
