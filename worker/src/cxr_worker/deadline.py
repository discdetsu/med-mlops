import signal
from collections.abc import Iterator
from contextlib import contextmanager
from types import FrameType


class JobTimeout(Exception):
    pass


@contextmanager
def deadline(seconds: float) -> Iterator[None]:
    """Raise JobTimeout in the main thread after `seconds`.

    Celery's time limits need the prefork pool; the solo pool runs jobs in the main thread,
    where SIGALRM can interrupt them instead.
    """

    def _expire(signum: int, frame: FrameType | None) -> None:
        raise JobTimeout(f"job exceeded {seconds:g}s")

    previous = signal.signal(signal.SIGALRM, _expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
