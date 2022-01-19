import tensorflow as tf
import tensorflow.keras as keras


class PrimitiveModelEnd(keras.layers.Layer):
    def call(self, inputs):
        return inputs

    def get_config(self):
        return {
            "name": self.name
        }


custom_objects = {
    "PrimitiveModelEnd": PrimitiveModelEnd,
}


if __name__ == '__main__':
    print("nothing")
