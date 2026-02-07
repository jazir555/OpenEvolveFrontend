"""sources package."""

from .test_aws import TestAws
from .test_azure import TestAzure
from .test_exceptions import TestExceptions
from .test_gcs import TestGcs
from .test_git import TestGit
from .test_google_drive import TestGoogleDrive
from .test_hf import TestHf
from .test_local import TestLocal
from .test_source_discriminator import TestSourceDiscriminator
from .test_web import TestWeb

__all__ = ['test_aws', 'test_azure', 'test_exceptions', 'test_gcs', 'test_git', 'test_google_drive', 'test_hf', 'test_local', 'test_source_discriminator', 'test_web']
