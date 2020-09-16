from amanda.namespace import Namespace


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


class MismatchNamespaceError(Exception):
    """
    Raised when the namespace is mismatched.
    """

    def __init__(self, expect: Namespace, actual: Namespace) -> None:
        return super().__init__(f"expect: {expect}, actual: {actual}")
