class QACLIError(Exception):
    """General error that I might want to track in the application."""

    pass


class QACLIForgivableError(Exception):
    """This question will ask the user to try again!"""

    pass
