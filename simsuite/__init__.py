import os

__DEBUG = str(os.getenv("DEBUG", "0")).lower() in ("1", "true", "yes", "on")


def dprint(*args, **kwargs):
    """Debug print function that can be easily toggled on/off."""
    if __DEBUG:
        print(*args, **kwargs)
