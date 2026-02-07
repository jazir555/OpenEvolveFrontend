"""tests package."""

from .fixtures import Fixtures
from .test_basic import TestBasic
from .test_chunked import TestChunked
from .test_chunk_text import TestChunkText
from .test_clustering import TestClustering
from .test_configs import TestConfigs

__all__ = ['fixtures', 'test_basic', 'test_chunked', 'test_chunk_text', 'test_clustering', 'test_configs']
