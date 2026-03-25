"""Attribute interfaces and adapters."""
from .base import AttributeExtractor
from .factory import build_attribute_extractor
from .person_attributes import PersonAttributeExtractor
from .simple_model import AttributeBaseline
__all__ = [
    "AttributeBaseline",
    "AttributeExtractor",
    "PersonAttributeExtractor",
    "build_attribute_extractor",
]
