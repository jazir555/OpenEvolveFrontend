"""
Aiohttp Compatibility Shim for OpenEvolve Knowledge Engine

This module patches aiohttp to be compatible with litellm (used by dspy/kg-gen).
aiohttp 3.9+ removed several timeout error classes but litellm still expects them.

This module should be imported BEFORE any module that imports dspy or litellm.
"""

try:
    import aiohttp
    
    # ConnectionTimeoutError -> ServerTimeoutError
    if not hasattr(aiohttp, 'ConnectionTimeoutError'):
        aiohttp.ConnectionTimeoutError = aiohttp.ServerTimeoutError
    
    # SocketTimeoutError -> ServerTimeoutError  
    if not hasattr(aiohttp, 'SocketTimeoutError'):
        aiohttp.SocketTimeoutError = aiohttp.ServerTimeoutError
        
except ImportError:
    pass  # aiohttp not installed
