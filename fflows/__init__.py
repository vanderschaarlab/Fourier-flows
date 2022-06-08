import sys

from . import logger  # noqa: F401
from .sequential_flows import FourierFlow, RealNVP

logger.add(sink=sys.stderr, level="CRITICAL")
