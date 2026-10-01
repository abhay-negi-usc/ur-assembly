"""relay -- relay K1, toggled by 'm'. Firmware: firmware/relay.cpp.

The board answers "Motor On" / "Motor Off". Any board reset switches it off, including every
disconnect (unless the connection was opened with --latch).
"""

from ..registry import Module

TOGGLE = 'm'


class Relay:
    def __init__(self, board, settings):
        self.board = board

    def toggle(self):
        """Toggle relay K1. Returns True if it ended up ON."""
        final, _ = self.board.exchange(TOGGLE, lambda ln: ln.startswith('Motor'))
        print(f"{self.board.tag}motor: {final}")
        return final == 'Motor On'


MODULE = Module('relay', 'relay K1 (firmware/relay.cpp)', device=Relay)


@MODULE.command()
def cmd_motor(s):
    """Toggle relay K1 on or off.

    The relay is switched off by any board reset, including every disconnect (unless the
    connection was opened with --latch)."""
    s.dev('relay').toggle()
