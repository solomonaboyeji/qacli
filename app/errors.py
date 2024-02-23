# error codes
ERROR_CODES = {
    "limited_knowledge_error": {
        "name": "Limited Knowledge Error",
        "description": "The model does not have enough knowledge.",
        "code": "000",
    }
    # other codes here
}


class QACLIError(Exception):
    """General error that I might want to track in the application."""

    pass


class QACLIForgivableError(Exception):
    """This question will ask the user to try again!"""

    pass
