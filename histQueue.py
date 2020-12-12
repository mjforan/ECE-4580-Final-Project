class HistoryQueue:
    def __init__(self, max_size, function):
        self.max_size = max_size
        self.queue = []
        self.function = function
    def push(self, item):
        if len(self.queue) >= self.max_size:
            self.pop()
        self.queue.append(item)

    def pop(self):
        if len(self.queue) > 0:
            self.queue.pop(0)

    def getFiltered(self):
        return self.function(self.queue)
