"""
Aiohttp Compatibility Shim for OpenEvolve Knowledge Engine

This module patches aiohttp to be compatible with litellm (used by dspy/kg-gen).
aiohttp 3.9+ removed several timeout error classes but litellm still expects them.

This module should be imported BEFORE any module that imports dspy or litellm.
"""

import logging
logger = logging.getLogger(__name__)

try:
    import aiohttp

    # ConnectionTimeoutError -> ServerTimeoutError
    if not hasattr(aiohttp, 'ConnectionTimeoutError'):
        aiohttp.ConnectionTimeoutError = aiohttp.ServerTimeoutError

    # SocketTimeoutError -> ServerTimeoutError
    if not hasattr(aiohttp, 'SocketTimeoutError'):
        aiohttp.SocketTimeoutError = aiohttp.ServerTimeoutError

    logger.debug("aiohttp compatibility patch applied successfully")

except ImportError:
    # aiohttp not installed - create dummy classes to prevent AttributeError
    logger.warning("aiohttp not installed - compatibility shim disabled. Install with: pip install aiohttp")

    # Create a dummy module to prevent AttributeError when code tries to access these classes
    class DummyTimeoutError(Exception):
        """Dummy timeout error for when aiohttp is not installed"""
        pass

    class DummyAiohttp:
        """Dummy aiohttp module for when aiohttp is not installed"""
        ServerTimeoutError = DummyTimeoutError
        ConnectionTimeoutError = DummyTimeoutError
        SocketTimeoutError = DummyTimeoutError

    import sys
    sys.modules['aiohttp_compatibility'] = DummyAiohttp()

