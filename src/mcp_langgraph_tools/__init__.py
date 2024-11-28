"""Mcp Langgraph Tools."""

from __future__ import annotations

import os
import warnings

from langchain._api import LangChainDeprecationWarning
from langchain_core._api import LangChainBetaWarning

warnings.simplefilter("ignore", category=LangChainDeprecationWarning)
warnings.simplefilter("ignore", category=LangChainBetaWarning)


__author__ = "Paul Robello"
__credits__ = ["Paul Robello"]
__maintainer__ = "Paul Robello"
__email__ = "probello@gmail.com"
__version__ = "0.1.0"
__application_title__ = "Mcp Langgraph Tools"
__application_binary__ = "mcp_langgraph_tools"
__licence__ = "MIT"


os.environ["USER_AGENT"] = f"{__application_title__} {__version__}"


__all__: list[str] = [
    "__author__",
    "__credits__",
    "__maintainer__",
    "__email__",
    "__version__",
    "__application_binary__",
    "__licence__",
    "__application_title__",
]
