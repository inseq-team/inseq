import logging
import os
import pickle
from functools import wraps
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Cache location
DEFAULT_XDG_CACHE_HOME = "~/.cache"
XDG_CACHE_HOME = os.getenv("XDG_CACHE_HOME", DEFAULT_XDG_CACHE_HOME)
DEFAULT_INSEQ_HOME_CACHE = os.path.join(XDG_CACHE_HOME, "inseq")
INSEQ_HOME_CACHE = os.path.expanduser(os.getenv("INSEQ_HOME", DEFAULT_INSEQ_HOME_CACHE))

DEFAULT_INSEQ_ARTIFACTS_CACHE = os.path.join(INSEQ_HOME_CACHE, "artifacts")
INSEQ_ARTIFACTS_CACHE = Path(os.getenv("INSEQ_ARTIFACTS_CACHE", DEFAULT_INSEQ_ARTIFACTS_CACHE))


def cache_results(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
    @wraps(func)
    def cache_results_wrapper(
        cache_dir: str,
        cache_filename: str,
        save_cache: bool = True,
        overwrite_cache: bool = False,
        *args,
        **kwargs,
    ):
        cache_dir = os.path.expanduser(cache_dir)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        if os.path.exists(cache_filename) and not overwrite_cache:
            logger.info(f"Loading cached objects from {cache_filename}")
            with open(cache_filename, "rb") as f:
                cached = pickle.load(f)  # nosec
        else:
            logger.info(f"Cached objects not found in {cache_filename}. Computing...")
            cached = func(*args, **kwargs)
            if save_cache:
                with open(cache_filename, "wb") as f:
                    pickle.dump(cached, f)
        return cached

    return cache_results_wrapper
