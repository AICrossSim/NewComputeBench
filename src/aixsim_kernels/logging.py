from typing import Union
import logging

import colorlog

root_logger = logging.getLogger("aixsim_kernels")

_handler = colorlog.StreamHandler()
_handler.setFormatter(
    colorlog.ColoredFormatter("%(log_color)s%(levelname)s:%(name)s:%(message)s")
)
root_logger.addHandler(_handler)


def set_logging_verbosity(level: Union[int, str]):
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)

    root_logger.info(f"Logging verbosity set to {level}")
