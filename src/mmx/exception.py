class IrremovableOpError(Exception):
    """
    Raised when the op cannot be removed from graph
    since some other ops still depends on it.
    """
