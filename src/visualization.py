import matplotlib.pyplot as plt


plt.ion()

class Visualization(object):
    def __init__(self, mini, maxi):
        self._fig = plt.figure()
        self._critic_prediction_ax = self._fig.add_subplot(121)
        self._critic_prediction_ax.set_ylim([mini, maxi])
        self._policy_prediction_ax = self._fig.add_subplot(122)
        self._policy_prediction_ax.set_ylim([mini, maxi])
        self._critic_target_line = None
        self._policy_target_line = None
        self._shown = False

    def update_critic(self, target, prediction):
        if self._critic_target_line is None:
            self._critic_target_line, = self._critic_prediction_ax.plot(
                target, label='target'
            )
            self._critic_prediction_line, = self._critic_prediction_ax.plot(
                prediction, label='prediction'
            )
            self._critic_prediction_ax.legend()
            if not self._shown:
                self._fig.show()
            self._fig.canvas.flush_events()
        else:
            self._critic_target_line.set_ydata(target)
            self._critic_prediction_line.set_ydata(prediction)
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()

    def update_policy(self, target, prediction):
        if self._policy_target_line is None:
            self._policy_target_line, = self._policy_prediction_ax.plot(
                target, label='target'
            )
            self._policy_prediction_line, = self._policy_prediction_ax.plot(
                prediction, label='prediction'
            )
            self._policy_prediction_ax.legend()
            if not self._shown:
                self._fig.show()
            self._fig.canvas.flush_events()
        else:
            self._policy_target_line.set_ydata(target)
            self._policy_prediction_line.set_ydata(prediction)
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()

    def __del__(self):
        plt.close(self._fig)


if __name__ == '__main__':
    import time

    v = Visualization()
    v([1,2,3], [4, 5, 6])
    time.sleep(1)
