import numpy as np

a = np.random.normal(size=(10,32))
b = np.random.normal(size=(10,64))

for item1, item2 in zip(a,b):
    print(item1.size, item2.size)