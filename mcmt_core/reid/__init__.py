"""Re-identification interfaces and adapters."""
from .base import ReIDExtractor
from .mgn import MGNReIDExtractor
__all__ = ["ReIDExtractor", "MGNReIDExtractor"]
