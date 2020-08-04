import tensorflow as tf
import tensorflow.keras as keras


class ReConcatDense(keras.layers.Dense):
    def call(self, inputs):
        return tf.concat([inputs, super().call(inputs)], axis=-1)


custom_objects = {
    "ReConcatDense": ReConcatDense,
}


if __name__ == '__main__':
    a = keras.models.Sequential([ReConcatDense(12)])
    yaml = a.to_yaml()
    print(yaml)
    b = keras.models.model_from_yaml(yaml, custom_objects=custom_objects)
