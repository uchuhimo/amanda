import threading


class ThreadLocalStack(threading.local):
    """A thread-local stack."""

    def __init__(self):
        super(ThreadLocalStack, self).__init__()
        self.stack = []

    def push(self, item):
        self.stack.append(item)

    def pop(self):
        self.stack.pop()

    def top(self):
        if len(self.stack) != 0:
            return self.stack[-1]
        else:
            return None

    def __iter__(self):
        return reversed(self.stack).__iter__()