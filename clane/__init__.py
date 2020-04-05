import logging

from .manager import ContextManager

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


ContextManager.instance()
g = ContextManager.instance()
