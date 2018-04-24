"""Jello."""
from multiprocessing import Process, Queue
from os import getpid


def f(x, q):
    """Function."""
    print(getpid())
    val = x * x
    q.put(val)
    return q


if __name__ == '__main__':
    q = Queue()
    jobs = []
    output = []
    for i in range(10):
        p = Process(target=f, args=(i, q))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    for i in range(10):
        output.append(q.get())

    print(output)
