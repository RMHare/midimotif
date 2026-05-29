from __future__ import annotations


def detect_exact_repetition(a: list[int], b: list[int]) -> bool:
    return a == b


def detect_transposition(a: list[int], b: list[int]) -> int | None:
    if len(a) != len(b) or not a:
        return None
    shift = b[0] - a[0]
    for x, y in zip(a, b):
        if y - x != shift:
            return None
    return shift
