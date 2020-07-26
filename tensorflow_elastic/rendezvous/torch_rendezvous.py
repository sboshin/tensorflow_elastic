try:
    from urllib.parse import urlparse, urlunparse
except ImportError:
    from urlparse import urlparse, urlunparse

import six
import numbers
import os
from .constants import default_pg_timeout


_rendezvous_handlers = {}


def register_rendezvous_handler(scheme, handler):
    """Registers a new rendezvous handler.

    Before we can run collective algorithms, participating processes
    need to find each other and exchange information to be able to
    communicate. We call this process rendezvous.

    The outcome of the rendezvous process is a triplet containing a
    shared key/value store, the rank of the process, and the total
    number of participating processes.

    If none of the bundled rendezvous methods apply to your execution
    environment you can opt to register your own rendezvous handler.
    Pick a unique name and use the URL scheme to identify it when
    calling the `rendezvous()` function.

    Arguments:
        scheme (str): URL scheme to identify your rendezvous handler.
        handler (function): Handler that is invoked when the
            `rendezvous()` function is called with a URL that uses
            the corresponding scheme. It must be a generator function
            that yields the triplet.
    """
    global _rendezvous_handlers
    if scheme in _rendezvous_handlers:
        raise RuntimeError(
            "Rendezvous handler for {}:// already registered".format(scheme)
        )
    _rendezvous_handlers[scheme] = handler


def rendezvous(url, rank=-1, world_size=-1, **kwargs):
    if not isinstance(url, six.string_types):
        raise RuntimeError("`url` must be a string. {}: {}".format(type(url), url))

    if not isinstance(rank, numbers.Integral):
        raise RuntimeError("`rank` must be an integer. {}".format(rank))

    if not isinstance(world_size, numbers.Integral):
        raise RuntimeError("`world_size` must be an integer. {}".format(world_size))

    # Append node-specific arguments.
    result = urlparse(url)
    if rank != -1 or world_size != -1:
        query_dict = dict(
            pair.split("=") for pair in filter(None, result.query.split("&"))
        )
        assert (
            "rank" not in query_dict and "world_size" not in query_dict
        ), "The url: {url} has node-specific arguments(rank, world_size) already.".format(
            url=url
        )
        if rank != -1:
            query_dict["rank"] = rank
        if world_size != -1:
            query_dict["world_size"] = world_size

        result = result._replace(
            query="{}".format("&".join(["{}={}".format(k, v) for k, v in query_dict.items()]))
        )
        url = urlunparse(result)

    if result.scheme not in _rendezvous_handlers:
        raise RuntimeError("No rendezvous handler for {}://".format(result.scheme))
    return _rendezvous_handlers[result.scheme](url, **kwargs)


def _rendezvous_error(msg):
    return ValueError("Error initializing torch.distributed using " + msg)

