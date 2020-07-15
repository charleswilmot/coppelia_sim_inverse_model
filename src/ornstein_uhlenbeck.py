import tensorflow as tf


class OUProcess(tf.Module):
  """A zero-mean Ornstein-Uhlenbeck process."""

  def __init__(self,
               shape,
               damping=0.15,
               stddev=0.2,
               seed=None,
               scope='ornstein_uhlenbeck_noise'):
    """A Class for generating noise from a zero-mean Ornstein-Uhlenbeck process.
    The Ornstein-Uhlenbeck process is a process that generates temporally
    correlated noise via a random walk with damping. This process describes
    the velocity of a particle undergoing brownian motion in the presence of
    friction. This can be useful for exploration in continuous action
    environments with momentum.
    The temporal update equation is:
    `x_next = (1 - damping) * x + N(0, std_dev * sqrt(damping * (2 - damping)))`
    Args:
      initial_value: Initial value of the process.
      damping: The rate at which the noise trajectory is damped towards the
        mean. We must have 0 <= damping <= 1, where a value of 0 gives an
        undamped random walk and a value of 1 gives uncorrelated Gaussian noise.
        Hence in most applications a small non-zero value is appropriate.
      stddev: Standard deviation of the Gaussian component.
      seed: Seed for random number generation.
      scope: Scope of the variables.
    """
    super(OUProcess, self).__init__()
    self._damping = damping
    self._stddev = stddev
    self._seed = seed
    with tf.name_scope(scope):
      initial_value = tf.random.normal(
        shape=shape,
        stddev=stddev,
        dtype=tf.float32,
      )
      self._x = tf.compat.v2.Variable(
          initial_value=initial_value, trainable=False)

  def __call__(self):
    noise = tf.random.normal(
        shape=self._x.shape,
        stddev=self._stddev * tf.sqrt(self._damping * (2 - self._damping)),
        dtype=self._x.dtype,
        seed=self._seed)
    return self._x.assign((1. - self._damping) * self._x + noise)
