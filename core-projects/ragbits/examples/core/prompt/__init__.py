"""prompt package."""

from .multimodal_with_few_shots import MultimodalWithFewShots
from .multimodal_with_image import MultimodalWithImage
from .multimodal_with_pdf import MultimodalWithPdf
from .text import Text
from .text_with_few_shots import TextWithFewShots

__all__ = ['multimodal_with_few_shots', 'multimodal_with_image', 'multimodal_with_pdf', 'text', 'text_with_few_shots']
