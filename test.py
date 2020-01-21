


import numpy as np
import statistics
import time

a = np.random.randint(2, size=(1,10))
b = np.random.randint(2, size=(1000000,10))
hamm = 0
t = time.time()
# for x in b:
#     value = np.count_nonzero(a != x)
#     hamm+=value
# print(hamm)
# print(time.time() - t)

# hamm = 0
# t = time.time()
# hamming = np.array([np.count_nonzero(a != x) for x in b])
# hamm = np.sum(hamming)
# print(hamm)
# print(time.time() - t)

hamm = 0
t = time.time()
hamming = [np.count_nonzero(a != x) for x in b]
hamm = sum(hamming)
print(hamm/len(hamming))
print(time.time() - t)
