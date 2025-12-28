import sys

import logging
logging.basicConfig(
    level=logging.WARNING,  # Show warning and above
    format="[%(levelname)s]: %(message)s"
)
logger = logging.getLogger(__name__)

