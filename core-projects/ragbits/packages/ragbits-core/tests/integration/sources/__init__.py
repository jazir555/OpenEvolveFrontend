"""sources package."""

from .test_git import TestGit
from .test_hf import TestHf
from .test_s3 import TestS3

__all__ = ['test_git', 'test_hf', 'test_s3']
