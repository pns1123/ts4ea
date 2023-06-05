import msgpack
import msgpack_numpy as m
import numpy as np

x = np.random.rand(5)
print(x)
x_enc = msgpack.packb(x, default=m.encode)
print(x_enc)
print(type(x_enc))
x_enc_str = str(x_enc)
print(str.encode(x_enc_str))
print(x_enc)
x_rec = msgpack.unpackb(str.encode(x_enc_str), object_hook=m.decode)
print(x_rec)
