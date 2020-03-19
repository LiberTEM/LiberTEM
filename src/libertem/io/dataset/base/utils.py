class FileTree(object):
    def __init__(self, low, high, value, idx, left, right):
        self.low = low
        self.high = high
        self.value = value
        self.idx = idx
        self.left = left
        self.right = right

    @classmethod
    def make(cls, files):
        """
        build a balanced binary tree by bisecting the files list
        """

        def _make(files):
            if len(files) == 0:
                return None
            mid = len(files) // 2
            idx, value = files[mid]

            return FileTree(
                low=value.start_idx,
                high=value.end_idx,
                value=value,
                idx=idx,
                left=_make(files[:mid]),
                right=_make(files[mid + 1:]),
            )
        return _make(list(enumerate(files)))

    def search_start(self, value):
        """
        search a node that has start_idx <= value && end_idx > value
        """
        if self.low <= value and self.high > value:
            return self.idx, self.value
        elif self.low > value:
            return self.left.search_start(value)
        else:
            return self.right.search_start(value)
