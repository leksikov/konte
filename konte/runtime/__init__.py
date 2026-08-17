"""Process-wide machinery: configuration, chat clients, and the serving cache.

The chat clients and the cache are reached through their own modules. Importing
them here would make every settings lookup pull in a chat client, and the cache
holds projects, which are assembled from every layer above this one.
"""

from konte.runtime.settings import Settings, settings

__all__ = [
    "Settings",
    "settings",
]
