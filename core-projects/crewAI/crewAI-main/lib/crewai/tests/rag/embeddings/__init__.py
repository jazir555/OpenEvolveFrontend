"""embeddings package."""

from .test_backward_compatibility import TestBackwardCompatibility
from .test_embedding_factory import TestEmbeddingFactory
from .test_factory_azure import TestFactoryAzure

__all__ = ['test_backward_compatibility', 'test_embedding_factory', 'test_factory_azure']
