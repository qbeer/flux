"""
Logging utilities
"""


def log_message(message: str) -> None:
    print('[Flux] {}'.format(message))


def log_warning(message: str) -> None:
    print('[Flux] Warning: {}'.format(message))
