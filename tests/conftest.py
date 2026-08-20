import os

os.environ.setdefault("FASTMCP_DECORATOR_MODE", "object")

# --- aioresponses / aiohttp 3.14 compatibility shim -------------------------
# aiohttp 3.14 made `stream_writer` a required keyword-only argument of
# ClientResponse.__init__. aioresponses 0.7.9 does not pass it, so every mocked
# request raises TypeError before the test body runs. The fix is sitting
# unmerged upstream (pnuckowski/aioresponses#288) on an unmaintained project,
# so patch it here rather than hold aiohttp back: 3.13.5 is affected by
# GHSA-cq5v-8q36-5273 (high, out-of-bounds heap read in the C response parser)
# plus three lower-severity advisories, all fixed by 3.14.3.
#
# Remove this once aioresponses ships aiohttp 3.14 support.

import inspect  # noqa: E402
from unittest.mock import Mock  # noqa: E402

import aiohttp  # noqa: E402
import aioresponses.core  # noqa: E402

if "stream_writer" in inspect.signature(aiohttp.ClientResponse).parameters:
    _original_build_response = aioresponses.core.RequestMatch._build_response

    def _build_response(self, *args, **kwargs):
        real_init = aiohttp.ClientResponse.__init__

        def init(response_self, *a, **kw):
            kw.setdefault("stream_writer", Mock(output_size=0))
            return real_init(response_self, *a, **kw)

        aiohttp.ClientResponse.__init__ = init
        try:
            return _original_build_response(self, *args, **kwargs)
        finally:
            aiohttp.ClientResponse.__init__ = real_init

    aioresponses.core.RequestMatch._build_response = _build_response
