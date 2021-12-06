from timeit import default_timer


class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.timer = default_timer

    def __enter__(self):
        self.start = self.timer()
        return self

    def __exit__(self, *args):
        end = self.timer()
        self.elapsed_secs = end - self.start
        self.elapsed = self.elapsed_secs * 1000  # millisecs
        if self.verbose:
            print(f"elapsed time: {self.elapsed} ms")


def main():
    with Timer(verbose=True) as t:
        for i in range(1000):
            j = i * i
    print(t.elapsed)


if __name__ == "__main__":
    main()
