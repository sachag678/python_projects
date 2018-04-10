"""Multihread example."""
import threading


class MyThread(threading.Thread):
    """Threading class."""

    def __init__(self, threadID, n):
        """Initilaize."""
        threading.Thread.__init__(self)
        self.threadId = threadID
        self.n = n

    def run(self):
        """Run."""
        print("%s: %d" % ("Thread", self.threadId))
        self.factorial(self.n)

    def factorial(self, n):
        """Calculate factorial."""
        global threadId
        if n < 1:   # base case
            return 1
        else:
            return_number = n * self.factorial(n - 1)  # recursive call
            print(str(n) + '! = ' + str(return_number))
            return return_number

thread1 = MyThread(1, 5)
thread2 = MyThread(2, 4)

thread1.start()
thread2.start()
thread1.join()
thread2.join()
