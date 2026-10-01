"""Errors and argument parsing shared by the framework and every module."""

#  Longest timed thing anything accepts, in seconds -- a wait, a run, a ramp. The screwdrive
#  firmware enforces the same limit (maxRunMs in firmware/screwdrive.cpp).
MAX_RUN_S = 3600


class ToolChangerError(RuntimeError):
    pass


class Interrupted(ToolChangerError):
    """The operator pressed q: anything moving has been stopped and the command abandoned."""


def parse_int(value, limit, what, unit):
    """A whole signed number in -limit..limit, or ToolChangerError before anything is sent.

    Checked here rather than left to the board, which would clamp 600 to 100 and run the
    motor flat out on a typo. Floats and words are refused, not rounded or read as 0."""
    if isinstance(value, bool) or value is None:
        raise ToolChangerError(f'{what} wants a whole {unit}, not {value!r}')
    if isinstance(value, str):
        try:
            value = int(value.strip())
        except ValueError:
            raise ToolChangerError(
                f'{what} wants a whole {unit} from -{limit} to {limit}, got {value!r}')
    if not isinstance(value, int):
        raise ToolChangerError(f'{what} wants a whole {unit}, got {value!r}')
    if not -limit <= value <= limit:
        raise ToolChangerError(f'{what} {value} is out of range -{limit}..{limit} {unit}')
    return value


def parse_ms(seconds, what='duration'):
    """A duration in seconds -> whole ms, refused unless 1 ms .. MAX_RUN_S."""
    try:
        if isinstance(seconds, bool):
            raise ValueError
        s = float(seconds)
    except (TypeError, ValueError):
        raise ToolChangerError(f'{what} wants a time in seconds, got {seconds!r}')
    ms = round(s * 1000) if s == s else 0       # NaN is out of range, not a crash
    if not 1 <= ms <= MAX_RUN_S * 1000:
        raise ToolChangerError(f'{what} {seconds!r} s is out of range (0.001..{MAX_RUN_S} s)')
    return ms
