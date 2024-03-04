import ray
import time
import numpy as np

if __name__ == '__main__':
    a = [(50, 4), (2, 7), (2, 5)]
    print(np.mean(a, axis=0))
    print(np.var(a, axis=0))