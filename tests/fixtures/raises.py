"""CLI test model whose build() fails inside a helper (for trimmed-traceback tests)."""
from mech import Assembly


def _arm_length(scale):
    return 40.0 / scale  # ZeroDivisionError for scale=0


def build(scale=0.0) -> Assembly:
    asm = Assembly("raises")
    _arm_length(scale)
    return asm
