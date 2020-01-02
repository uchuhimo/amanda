class IrremovableOpError(Exception):
    """
    Raised when the op cannot be removed from graph
    since some other ops still depends on it.
    """


class OpMappingError(Exception):
    """
    Raised when an op cannot be mapped to another namespace.
    """

    def __init__(self, graph, op):
        self.graph = graph
        self.op = op
