import cobweb
from cobweb import CobwebNode, CobwebTree
import time

if __name__ == "__main__":

    s = time.time()
    for i in range(10000000):
        _ = CobwebTree()
    print(time.time() - s)