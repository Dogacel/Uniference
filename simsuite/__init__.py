import os

__DEBUG = bool(os.getenv("DEBUG", False))


def dprint(*args, **kwargs):
    """Debug print function that can be easily toggled on/off."""
    if __DEBUG:
        print(*args, **kwargs)
