import tensorflow as tf
from tensorflow import keras


class Model(keras.Model):
    def __init__(self, type, layers):
        super().__init__()
        self.type = type
        if self.type == "simple":
            sizes_activations = [
                [eval(str(layers[n].size)), layers[n].activation]
                for n in sorted(layers)
            ]
            activ = sizes_activations[-1][1]
            sizes_activations[-1][1] = "None"
            self.concat = keras.layers.Concatenate()
            self.denses = [
                keras.layers.Dense(
                    size,
                    activation=eval(activation)
                )
                for size, activation in sizes_activations
            ]
            self.last_layer_activation = keras.layers.Activation(eval(activ))
        else:
            raise ValueError("Unrecognized model type ({})".format(self.type))

    def call(self, *inputs, logits=False):
        if self.type == "simple":
            states, goals = inputs
            goals = tf.cast(goals, tf.float32)
            outputs = self.concat([states, goals])
            for dense in self.denses:
                outputs = dense(outputs)
            if logits:
                return outputs
            return self.last_layer_activation(outputs)
        else:
            raise ValueError("Unrecognized model type ({})".format(self.type))
