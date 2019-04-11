import multiprocessing


class MultiCallback():

    def __init__(self, target):
        self.queue = multiprocessing.Queue()
        self.target = target
        self.worker = multiprocessing.Process(target=self._work,
                                              args=(self.queue,))
        self.worker.start()

    def call(self, *args, **kwargs):
        self.queue.put((args, kwargs))

    def _work(self, queue):
        while True:
            args, kwargs = queue.get()
            if args is None and kwargs is None:
                break
            self.target(*args, **kwargs)

    def finish(self):
        self.queue.put((None, None))
        self.worker.join()
