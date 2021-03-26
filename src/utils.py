import numpy as np
import pickle


INT_LENGTH_IN_BYTES = 4


class appendable_array_file:
    def __init__(self, path):
        self._file = open(path, 'ab')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._file.close()

    def append(self, arr):
        f = self._file
        if f.tell() == 0:
            shape = list(arr.shape)
            shape[0] = -1
            shape = tuple(shape)
            serialized_shape = pickle.dumps(shape)
            serialized_shape_length = len(serialized_shape).to_bytes(INT_LENGTH_IN_BYTES, byteorder='big')
            serialized_dtype = pickle.dumps(arr.dtype)
            serialized_dtype_length = len(serialized_dtype).to_bytes(INT_LENGTH_IN_BYTES, byteorder='big')
            f.write(serialized_shape_length)
            f.write(serialized_shape)
            f.write(serialized_dtype_length)
            f.write(serialized_dtype)
        f.write(arr.tobytes())


def load_appendable_array_file(path):
    with open(path, 'rb') as f:
        serialized_shape_length = f.read(INT_LENGTH_IN_BYTES)
        serialized_shape = f.read(int.from_bytes(serialized_shape_length, byteorder='big'))
        serialized_dtype_length = f.read(INT_LENGTH_IN_BYTES)
        serialized_dtype = f.read(int.from_bytes(serialized_dtype_length, byteorder='big'))
        shape = pickle.loads(serialized_shape)
        dtype = pickle.loads(serialized_dtype)
        return np.frombuffer(f.read(), dtype=dtype).reshape(shape)


if __name__ == '__main__':
    dtype = np.dtype([('az', np.float32), ('qs', np.int32, (3, 3))])
    a = np.zeros(shape=(2, 2), dtype=dtype)
    a[0, 0]['az'] = 123

    print(a)
    print("\n")

    with appendable_array_file("/tmp/test.dat") as f:
        f.append(a)
    with appendable_array_file("/tmp/test.dat") as f:
        f.append(a)
    b = load_appendable_array_file("/tmp/test.dat")
    print(b)
