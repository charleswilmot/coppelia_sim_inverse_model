import tensorflow as tf
import tensorflow.keras as keras


class ReConcatDense(keras.layers.Dense):
    def call(self, inputs):
        return tf.concat([inputs, super().call(inputs)], axis=-1)


class LinspaceInitializer(keras.initializers.Initializer):
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop

    def __call__(self, shape, dtype=tf.float32):
        return tf.cast(tf.linspace(self.start, self.stop, shape[0]), dtype)

    def get_config(self):  # To support serialization
        return {"start": self.start, "stop": self.stop}


class NormalNoise(keras.layers.Layer):
    def __init__(self, n_simulations, min_log_stddev, max_log_stddev, clip_min=None, clip_max=None):
        super().__init__(self)
        self._n_simulations = n_simulations
        self._min_log_stddev = min_log_stddev
        self._max_log_stddev = max_log_stddev
        self._clip_min = clip_min
        self._clip_max = clip_max
        self._on = self.add_weight(
            "on",
            dtype=tf.bool,
            trainable=False,
        )
        self._explore = self.add_weight(
            "explore",
            shape=self._n_simulations,
            dtype=tf.bool,
            trainable=False,
        )
        self.log_stddevs = self.add_weight(
            "log_stddevs",
            shape=[self._n_simulations],
            dtype=tf.float32,
            initializer=LinspaceInitializer(self._min_log_stddev, self._max_log_stddev),
            trainable=False,
        )

    def call(self, inputs):
        if self._on:
            noise = tf.random.normal(shape=tf.shape(inputs)) * tf.exp(self.log_stddevs)[:, tf.newaxis]
            noise = tf.where(self._explore[:, tf.newaxis], noise, tf.zeros_like(noise))
            before_clipping = inputs + noise
            if self._clip_min is not None and self._clip_max is not None:
                return tf.clip_by_value(before_clipping, self._clip_min, self._clip_max)
            else:
                return before_clipping
        else:
            return inputs

    def set_on(self, new):
        self._on.assign(new)

    def set_explore(self, new):
        self._explore.assign(new)

    def set_log_stddevs(self, new):
        self.log_stddevs.assign(new)

    def get_config(self):
        return {
            "n_simulations":self._n_simulations,
            "min_log_stddev":self._min_log_stddev,
            "max_log_stddev":self._max_log_stddev,
            "clip_min":self._clip_min,
            "clip_max":self._clip_max,
        }


custom_objects = {
    "ReConcatDense": ReConcatDense,
    "NormalNoise": NormalNoise,
}


if __name__ == '__main__':
    # a = keras.models.Sequential([ReConcatDense(12)])
    # yaml = a.to_yaml()
    # print(yaml)
    # b = keras.models.model_from_yaml(yaml, custom_objects=custom_objects)

    a = LinspaceInitializer(-10, 10)
    print(a(tf.constant([5])))

    n_simulations = 40
    min_log_stddev = -3
    max_log_stddev = 0
    clip_min = -1
    clip_max = 1

    a = keras.models.Sequential([NormalNoise(n_simulations, min_log_stddev, max_log_stddev, clip_min=clip_min, clip_max=clip_max)])
    a.layers[0].set_on(True)
    a.layers[0].set_explore(tf.ones(40, dtype=tf.bool))
    print(a(tf.zeros(shape=[40, 2])))
    a.layers[0].set_on(False)
    print(a(tf.zeros(shape=[140, 2])))

    yaml = a.to_yaml()
    print(yaml)
    b = keras.models.model_from_yaml(yaml, custom_objects=custom_objects)
