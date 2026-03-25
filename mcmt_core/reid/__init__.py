"""Re-identification interfaces and adapters."""
from .base import ReIDExtractor
from .factory import build_reid_extractor
from .mgn import MGNReIDExtractor
from .mgn_model import MGN
__all__ = ["ReIDExtractor", "build_reid_extractor", "MGN", "MGNReIDExtractor"]
