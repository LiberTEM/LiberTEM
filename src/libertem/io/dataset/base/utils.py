import typing


NodeType = typing.Union[None, 'FileTree']


class FileTree:
    """
    Construct a FileTree node

    Parameters
    ----------

    low
        First frame contained in this file

    high
        First index of the next file

    value
        The corresponding file object

    idx
        The index of the file object in the fileset

    left
        Nodes with a lower low

    right
        Nodes with a higher low
    """
    def __init__(self, low: int, high: int, value: typing.Any, idx: int,
                 left: NodeType, right: NodeType):
        if low >= high:
            raise ValueError("low should be < high")
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

    def __str__(self):
        return self.to_string()

    def to_string(self, depth=0):
        padsingle = 4 * ' '
        pad = (depth * padsingle)
        str_left = (self.left is not None and self.left.to_string(depth + 1) or 'None')
        str_right = (self.right is not None and self.right.to_string(depth + 1) or 'None')
        str_self = f"(d={depth} l={self.low}, i={self.idx}, h={self.high})"
        return f"{pad}{str_left}\n{pad}{str_self}\n{pad}{str_right}"
