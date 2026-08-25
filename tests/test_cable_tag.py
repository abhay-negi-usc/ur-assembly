"""Coloured-tag cable selection.

The matcher is pure numpy, so the colour decision -- the part that can send the gripper to the
wrong cable -- is exercised on synthetic images here rather than only on the bench.
"""

import numpy as np
import pytest

from urlab.perception.tag import NAMED_HUES, TagMatcher, hue_distance, rgb_to_hsv


def patch(rgb, shape=(40, 40)):
    """A solid-colour image."""
    img = np.zeros(shape + (3,), dtype=np.uint8)
    img[:] = rgb
    return img


def test_rgb_to_hsv_matches_colorsys():
    """Hand-rolled to keep this module free of OpenCV, so it has to agree with the stdlib."""
    import colorsys
    rng = np.random.default_rng(0)
    px = rng.integers(0, 256, size=(200, 3)).astype(np.uint8)
    h, s, v = rgb_to_hsv(px)
    for i in range(len(px)):
        eh, es, ev = colorsys.rgb_to_hsv(*(px[i] / 255.0))
        assert abs(hue_distance(h[i], eh * 360.0)) < 1e-6, f'hue at {px[i]}'
        assert abs(s[i] - es) < 1e-9 and abs(v[i] - ev) < 1e-9


def test_red_wraps_around_the_hue_seam():
    """Red sits on the 0/360 seam, so a plain abs(h - 0) test misses half of it -- and red is the
    first colour anyone reaches for when tagging something."""
    m = TagMatcher('red')
    assert m.mask(patch((255, 0, 0))).all(), 'pure red'
    assert m.mask(patch((255, 40, 0))).all(), 'just above the seam'
    assert m.mask(patch((255, 0, 40))).all(), 'just below the seam (hue ~351)'
    assert not m.mask(patch((0, 255, 0))).any(), 'green is not red'
    assert hue_distance(359.0, 0.0) == pytest.approx(1.0)


def test_grey_and_shadow_are_not_a_colour():
    """THE FALSE-POSITIVE THAT MATTERS. Every grey pixel has some hue, and near-black hue is
    numerical noise -- so an unguarded hue test turns a shiny dark cable into a red tag. Saturation
    and value must gate before hue is consulted."""
    m = TagMatcher('red')
    for name, px in (('mid grey', (128, 128, 128)), ('white', (250, 250, 250)),
                     ('near black', (6, 2, 2)), ('dark desaturated red', (40, 30, 30))):
        assert not m.mask(patch(px)).any(), f'{name} must not read as a tag'


def test_the_score_counts_that_cables_own_matching_pixels():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :] = (30, 30, 30)
    img[0:20, :] = (255, 0, 0)                      # 2000 red pixels
    where = np.zeros((100, 100), dtype=bool)
    where[0:50, :] = True                           # ...all of them inside this region
    m = TagMatcher('red', {'min_pixels': 10})
    assert m.score(img, where) == 2000
    # Red OUTSIDE the region cannot be credited to it -- that is what stops one cable's tag, or a
    # red object on the bench, from selecting a different cable.
    elsewhere = np.zeros((100, 100), dtype=bool)
    elsewhere[60:100, :] = True
    assert m.score(img, elsewhere) == 0.0


def test_a_tiny_patch_of_colour_cannot_outrank_a_real_tag():
    """The reason a COUNT is the right score. As a fraction, 3 pixels of which 2 match reads as
    67% and beats a real tag; as a count it is 2, which is what it is worth."""
    img = patch((255, 0, 0), (10, 10))
    where = np.zeros((10, 10), dtype=bool)
    where[0, 0:3] = True
    m = TagMatcher('red', {'min_pixels': 300})
    assert m.score(img, where) == 3
    assert not m.passes(m.score(img, where))


def test_no_tag_configured_is_disabled_not_a_colour_called_none():
    """yaml parses bare `none` as the STRING 'none' (only null/~ are None), so a config saying
    "no tag" would become a colour named "none" if it were not normalised."""
    for spec in (None, 'none', 'None', 'off', '', 'null'):
        m = TagMatcher(spec)
        assert not m.enabled, f'{spec!r} must mean NO tag'
        assert m.score(patch((255, 0, 0)), np.ones((40, 40), bool)) == 0
        assert not m.passes(10 ** 6), 'a disabled matcher can never select'
    assert TagMatcher('red').enabled


def test_colour_names_and_explicit_hues_are_both_accepted():
    assert TagMatcher('red').hue == 0.0
    assert TagMatcher('GREEN').hue == NAMED_HUES['green']
    assert TagMatcher(210).hue == 210.0
    assert TagMatcher('210').hue == 210.0
    with pytest.raises(ValueError, match='not a known colour'):
        TagMatcher('reddish')


def _cables(scores):
    return [{'tag_score': s, 'tag_pass': None, 'size': 100 - i}
            for i, s in enumerate(scores)]


def test_selection_is_automatic_only_when_exactly_one_cable_is_tagged():
    """THE RULE. Zero over the threshold means the tag was not seen; two or more means it is not
    distinguishing them. Both must fall back to the prompt -- an auto-pick that is wrong sends the
    gripper somewhere real, and the threshold exists to make that failure loud, not silent."""
    from urlab.skills.ground_pick import GroundPlaneScanner

    m = TagMatcher('red', {'min_pixels': 300})
    for scores, expect, why in (
            ([1400, 20, 0], 0, 'exactly one over -> take it'),
            ([20, 0], None, 'none over -> ask'),
            ([1400, 900], None, 'two over -> ask, the tag is not distinguishing them'),
            ([300], 0, 'exactly at the threshold counts as over'),
            ([299], None, 'just under does not')):
        cables = _cables(scores)
        for c in cables:
            c['tag_pass'] = m.passes(c['tag_score'])
        scanner = GroundPlaneScanner.__new__(GroundPlaneScanner)   # no camera/robot needed
        scanner.tag = m
        scanner.labeled_path = '/tmp/x.png'
        assert scanner._auto_pick(cables) == expect, why

    # With no tag configured nothing is ever auto-selected, whatever the scores say.
    scanner = GroundPlaneScanner.__new__(GroundPlaneScanner)
    scanner.tag = TagMatcher(None)
    scanner.labeled_path = '/tmp/x.png'
    assert scanner._auto_pick(_cables([9000, 0])) is None


def test_the_bnc_carries_a_red_tag_and_the_others_carry_none():
    """The tag lives in cables.yaml so one marker is described in one place; check the file says
    what the app will read, including that `none` does not survive as a colour."""
    import os

    import yaml

    from urlab import config as urconfig
    with open(os.path.join(urconfig.CONFIG_DIR, 'cables.yaml')) as f:
        db = yaml.safe_load(f)
    cables = db['cables']
    assert str(cables['bnc']['tag_color']).lower() == 'red'
    for name in ('banana', 'hose', 'c13'):
        assert 'tag_color' in cables[name], f'{name} must say so explicitly, not by omission'
        assert not TagMatcher(cables[name]['tag_color']).enabled, f'{name} must have NO tag'
    assert TagMatcher(cables['bnc']['tag_color']).enabled


def test_the_cable_profile_normalises_the_tag_onto_the_config():
    """apply_cable_profile is what carries tag_color from cables.yaml to cable_tag.color."""
    from urlab import config as urconfig
    for cable, expect in (('bnc', 'red'), ('banana', None)):
        cfg = urconfig.load('bnc_assembly', [f'cable={cable}'])
        assert cfg.get_path('cable_tag.color') == expect, cable
        assert TagMatcher(cfg.get_path('cable_tag.color')).enabled is (expect is not None)


def test_cables_are_reordered_best_match_first_and_the_image_is_renumbered():
    """"Number them in order of whichever best matches the tag colour."

    The numbers drawn on the saved image ARE the numbers the user is asked to choose between, so
    re-ordering the list without re-labelling the picture would be worse than not ranking at all.
    This checks both halves: the order, and that the detector was asked to redraw it."""
    from urlab.skills.ground_pick import GroundPlaneScanner

    h = w = 60
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:] = (30, 30, 30)
    rgb[0:10, 0:40] = (255, 0, 0)          # a red band over cable B's rows only

    def band(r0, r1):
        m = np.zeros((h, w), dtype=bool)
        m[r0:r1, :] = True
        return m

    # A is the BIGGER component (so detection order puts it first) but carries no red.
    cables = [{'junction': (10.0, 30.0, 0.0), 'ends': [], 'cable_mask': band(20, 40), 'size': 900},
              {'junction': (5.0, 20.0, 0.0), 'ends': [], 'cable_mask': band(0, 10), 'size': 400}]

    class StubDetector:
        def __init__(self):
            self.labelled = None

        def detect_all(self, frame, max_cables):
            return list(cables)

        def label_cables(self, frame, cabs):
            self.labelled = [c['size'] for c in cabs]

    class Frame:
        pass

    frame = Frame()
    frame.rgb = rgb

    s = GroundPlaneScanner.__new__(GroundPlaneScanner)
    s.detector = StubDetector()
    s.max_cables = 8
    s.labeled_path = '/tmp/x.png'
    s.tag = TagMatcher('red', {'min_pixels': 100})

    ranked = s._detect_ranked(frame)
    assert [c['size'] for c in ranked] == [400, 900], \
        'the tagged cable must come first even though it is the smaller component'
    assert ranked[0]['tag_score'] > ranked[1]['tag_score']
    assert ranked[0]['tag_pass'] and not ranked[1]['tag_pass']
    assert s.detector.labelled == [400, 900], \
        'the image must be renumbered to the ranked order, not the detection order'
    assert s._auto_pick(ranked) == 0

    # With no tag configured the order is left exactly as detected, and nothing is renumbered.
    s2 = GroundPlaneScanner.__new__(GroundPlaneScanner)
    s2.detector = StubDetector()
    s2.max_cables = 8
    s2.labeled_path = '/tmp/x.png'
    s2.tag = TagMatcher(None)
    assert [c['size'] for c in s2._detect_ranked(frame)] == [900, 400]
    assert s2.detector.labelled is None, 'no tag -> no re-labelling pass'


def test_the_engage_speed_comes_from_phase_scale_not_a_private_key():
    """assembly.engage.speed_mm_s is retired: one phase carrying a private mm/s that silently
    outranked its own phase scale meant the speed table did not describe the run."""
    import os

    from urlab import config as urconfig
    src = open(os.path.join(os.path.dirname(urconfig.__file__), 'apps',
                            'bnc_assembly.py'), encoding='utf-8').read()
    assert "v_mm_s = g_v * s_eng" in src, 'the engage rate must be the cap x its phase scale'
    assert "s_eng = float(scales.get('engage', scales.get('assemble', 1.0)))" in src, \
        'engage needs its own scale, falling back to assemble so old configs are unchanged'
    assert "float(en_speed) if en_speed is not None" not in src, 'the private override is gone'
    assert "is RETIRED" in src, 'a stale speed_mm_s must fail loudly, not be ignored'
    # The engage is servo-driven, and servo_l calls servoL directly -- it NEVER reads
    # arm.speed_scale. So a phase() call cannot pace this motion; all it would do is leave the arm
    # at engage's very low scale for every ordinary move that follows, until the next phase().
    # v_mm_s is the single control.
    assert "phase('engage')" not in src, (
        'engage must not set an arm speed scale: servo_l ignores it, and it would leak a very '
        'low scale onto every following move')
    assert 'speed.phase_scale.engage x speed.max_cartesian_translation_mm_s' in src, \
        'the resolved reference rate must be logged, so the speed is answerable from the log'

    cfg = urconfig.load('bnc_assembly')
    assert cfg.get_path('assembly.engage.speed_mm_s') is None, 'the key must be gone from the yaml'
    cap = float(cfg.get_path('speed.max_cartesian_translation_mm_s'))
    scales = cfg.get_path('speed.phase_scale') or {}
    assert 'engage' in scales, 'bnc_assembly must declare its own engage scale'
    assert abs(cap * float(scales['engage']) - 2.5) < 1e-9, \
        'the split must preserve the 2.5 mm/s the retired key asked for'
    # ...and it must be settable INDEPENDENTLY of assemble, which is the point of splitting it.
    assert float(scales['engage']) != float(scales['assemble'])


def test_enter_only_defaults_to_number_one_when_something_actually_scored():
    """With every cable at 0 px there is no evidence: the order is merely largest-first, so #1 is
    not a "top pick" and Enter must not take it. Requiring a number is the honest answer -- the
    alternative is grabbing whichever cable happens to be biggest."""
    import builtins

    from urlab.skills.ground_pick import GroundPlaneScanner

    def run(allow_default, keys):
        s = GroundPlaneScanner.__new__(GroundPlaneScanner)
        s.tag = TagMatcher('red', {'min_pixels': 300})
        s.labeled_path = '/tmp/x.png'
        s.z_step = 0.05
        it = iter(keys)
        real = builtins.input
        builtins.input = lambda *_a: next(it)
        try:
            return s._prompt(3, allow_default=allow_default)
        finally:
            builtins.input = real

    assert run(True, ['']) == 0, 'with a real match behind it, Enter takes the top pick'
    # All-zero: Enter is refused and the prompt keeps asking until a number arrives.
    assert run(False, ['', '', '2']) == 1, 'Enter must be ignored, then the typed number wins'
    # The other verbs still work with the default withdrawn.
    assert run(False, ['q']) is None
    assert run(False, ['n']) == 'new'


def test_the_scan_withdraws_the_default_when_no_cable_scored():
    """The caller decides: `allow_default` comes from whether ANY cable has a non-zero count."""
    import inspect

    from urlab.skills import ground_pick
    src = inspect.getsource(ground_pick.GroundPlaneScanner.scan)
    assert "ranked = any(int(c.get('tag_score') or 0) > 0 for c in cables)" in src
    assert 'allow_default=ranked' in src
