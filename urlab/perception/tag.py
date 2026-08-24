"""COLOURED TAG matching -- pick the cable that is wearing the marker you put on it.

WHY A TAG. The ground-plane scan detects every cable it can see and asks a human which one to
grasp. That is fine once, and tedious every run: the answer is always "the one I set up". A small
coloured band on the target cable turns that question into a measurement -- and, unlike matching on
shape or position, it survives the cable being moved, re-coiled, or lying next to identical ones.

WHAT IS SCORED. The FRACTION of a cable's own pixels that match the colour, not the presence of a
matching pixel anywhere. A single red pixel is noise; a band of them is a tag. And the fraction is
taken over the CABLE-side pixels of one junction, so a red object lying in the background, or on a
different cable, cannot lend its colour to this one.

WHY HSV, AND WHY SATURATION AND VALUE GATE FIRST. Hue is the only channel that means "what colour
is this", but it is meaningless where there is nothing to be coloured: at low saturation every grey
pixel has some hue, and at low value the hue of near-black is numerical noise. Shadows and specular
highlights on a glossy cable are exactly those two cases, and they are what turns an unguarded hue
test into a detector of shiny black cable. So a pixel must clear `min_saturation` and `min_value`
BEFORE its hue is looked at.

RED WRAPS. Hue is a circle and red sits on the seam at 0/360, so a naive `abs(h - 0) <= tol` misses
every pixel just below 360. Distances here are circular; that is not an edge case to handle later,
it is the default colour someone reaches for when tagging something.

Pure numpy on purpose -- no OpenCV -- so the matching can be tested on synthetic images in CI.
"""

import numpy as np

from .. import log as urlog

log = urlog.get('tag')

# Hue centres in degrees. These are the names someone writing a config will reach for; anything
# else can be given as a number (degrees) instead of a name.
NAMED_HUES = {
    'red': 0.0,
    'orange': 30.0,
    'yellow': 55.0,
    'green': 120.0,
    'cyan': 180.0,
    'blue': 225.0,
    'purple': 280.0,
    'violet': 280.0,
    'magenta': 300.0,
    'pink': 330.0,
}

# `tag_color: none` in yaml is the STRING 'none' (only null/~ parse as None), and a config that
# says "no tag" must not be read as a colour named "none".
NO_TAG = ('', 'none', 'null', 'nil', 'off', 'false')


def rgb_to_hsv(rgb):
    """Vectorised RGB -> (hue deg [0,360), saturation [0,1], value [0,1]).

    `rgb` is (..., 3), either uint8 0-255 or float 0-1. Written out rather than taken from cv2 so
    this module -- and therefore the tag matching -- has no OpenCV dependency and can be exercised
    on synthetic images in the test suite."""
    a = np.asarray(rgb)
    a = a.astype(np.float64) / 255.0 if a.dtype == np.uint8 else a.astype(np.float64)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx, mn = a.max(axis=-1), a.min(axis=-1)
    d = mx - mn
    h = np.zeros_like(mx)
    lit = d > 1e-12
    is_r = lit & (mx == r)
    is_g = lit & (mx == g) & ~is_r
    is_b = lit & ~is_r & ~is_g
    with np.errstate(invalid='ignore', divide='ignore'):
        h[is_r] = ((g - b)[is_r] / d[is_r]) % 6.0
        h[is_g] = ((b - r)[is_g] / d[is_g]) + 2.0
        h[is_b] = ((r - g)[is_b] / d[is_b]) + 4.0
    h = (h * 60.0) % 360.0
    s = np.where(mx > 1e-12, d / np.maximum(mx, 1e-12), 0.0)
    return h, s, mx


def hue_distance(h, centre):
    """Circular distance in degrees between hues -- so red at 359 is 1 deg from red at 0."""
    d = np.abs(np.asarray(h, dtype=float) - float(centre)) % 360.0
    return np.minimum(d, 360.0 - d)


class TagMatcher:
    """Scores how strongly a set of pixels wears a named colour.

    `color` is a name from NAMED_HUES, a hue in degrees, or one of NO_TAG / None for "no tag", in
    which case `enabled` is False and every score is 0.0 -- so a cable with no tag configured takes
    the ordinary manual path without the caller needing to special-case it."""

    def __init__(self, color=None, cfg=None):
        c = dict(cfg or {})
        self.hue_tol = float(c.get('hue_tolerance_deg', 18.0))
        self.min_sat = float(c.get('min_saturation', 0.35))
        self.min_val = float(c.get('min_value', 0.20))
        # The bar a cable must clear to be called TAGGED. A fraction, not a count, so it does not
        # change meaning with camera resolution or how close the arm has approached.
        self.min_fraction = float(c.get('min_fraction', 0.05))
        self.min_pixels = int(c.get('min_pixels', 40))
        self.name, self.hue = self._parse(color)
        self.enabled = self.hue is not None

    @staticmethod
    def _parse(color):
        if color is None:
            return None, None
        if isinstance(color, (int, float)) and not isinstance(color, bool):
            return f'{float(color):.0f}deg', float(color) % 360.0
        text = str(color).strip().lower()
        if text in NO_TAG:
            return None, None
        if text in NAMED_HUES:
            return text, NAMED_HUES[text]
        try:
            return f'{float(text):.0f}deg', float(text) % 360.0
        except ValueError:
            raise ValueError(
                f'tag_color {color!r} is not a known colour or a hue in degrees. '
                f'Known: {", ".join(sorted(NAMED_HUES))}; or write a number, or one of '
                f'{"/".join(NO_TAG[1:3])} for no tag.') from None

    def mask(self, rgb):
        """Boolean image of pixels that ARE the tag colour: saturated and bright enough for hue to
        mean anything, and within the hue tolerance."""
        h, s, v = rgb_to_hsv(rgb)
        return (s >= self.min_sat) & (v >= self.min_val) & \
            (hue_distance(h, self.hue) <= self.hue_tol)

    def score(self, rgb, where):
        """Fraction of the pixels in `where` that match. 0.0 if disabled or too few pixels to judge.

        A tiny region is refused rather than scored: 3 pixels of which 2 match is 67%, which would
        outrank a real tag, and a distant or clipped cable is exactly where that happens."""
        if not self.enabled or where is None:
            return 0.0
        where = np.asarray(where, dtype=bool)
        n = int(where.sum())
        if n < self.min_pixels:
            return 0.0
        return float(np.count_nonzero(self.mask(rgb) & where)) / float(n)

    def passes(self, score):
        return bool(self.enabled and score >= self.min_fraction)

    def describe(self):
        if not self.enabled:
            return 'no tag colour configured'
        return (f'tag {self.name} (hue {self.hue:.0f}+-{self.hue_tol:.0f} deg, '
                f'sat>={self.min_sat:.2f}, val>={self.min_val:.2f}), '
                f'selected at >={self.min_fraction * 100:.0f}% of the cable pixels')
