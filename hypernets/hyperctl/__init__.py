from typing import Optional

from .runtime import get_job_params


class Context:

    def __init__(self, executor_manager, batch):
        self.executor_manager = executor_manager
        self.batch = batch


_context = None


def set_context(c):
    global _context
    _context = c


def get_context() -> Optional[Context]:
    return _context
