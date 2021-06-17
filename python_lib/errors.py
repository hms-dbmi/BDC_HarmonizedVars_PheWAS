class ExtensionError(Exception):
    """
        Raised when the regex did not detect one of the file extensions to be written
    """


class NoVariablesError(Exception):
    """
        Raised when all variables are being filtered by quality checking
    """


class MappingError(Exception):
    """
        Raised when variables mapping are
    """